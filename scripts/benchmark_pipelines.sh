#!/usr/bin/env bash
#
# Bash port of benchmark_pipelines.ps1 — run this FROM THE LOAD-GENERATOR EC2
# INSTANCE in the same VPC as the API instance, targeting the API's PRIVATE IP.
# Generating load from a laptop over the public internet adds ~20ms+ of home
# Wi-Fi / WAN jitter to every sample and pollutes low-load latency comparisons.
#
# Usage:
#   ./scripts/benchmark_pipelines.sh http://<API_PRIVATE_IP>:8000 aws_results
#
set -u

TARGET_HOST="${1:-http://localhost:8000}"
RESULTS="${2:-results}"

STATS_DIR="$RESULTS/stats"
HISTORY_DIR="$RESULTS/history"
FAILURES_DIR="$RESULTS/failures"
EXCEPTIONS_DIR="$RESULTS/exceptions"
RUN_TIME="60s"
SPAWN_RATE=5
RECOVERY_SLEEP=15
REPEATS=5        # n=5 per cell lets per-level Mann-Whitney tests survive Holm correction
DRAIN_TIMEOUT=300

mkdir -p "$STATS_DIR" "$HISTORY_DIR" "$FAILURES_DIR" "$EXCEPTIONS_DIR"

ARCHS=(baseline synch asynch)
FILES=(tests/locust_baseline.py tests/locust_synch.py tests/locust_asynch.py)
LEVELS=(1 5 10 25 50)

total=$(( ${#ARCHS[@]} * ${#LEVELS[@]} * REPEATS ))
run_count=0

# Waits until the server reports an empty Celery queue. Requires 3 consecutive
# clean "drained" readings because Celery acks a task when execution STARTS —
# a lone zero reading can race with a task still running on a worker.
wait_queue_drain() {
    local deadline=$(( $(date +%s) + DRAIN_TIMEOUT ))
    local streak=0
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local body
        body=$(curl -sf --max-time 15 "$TARGET_HOST/queue/depth" 2>/dev/null || true)
        if echo "$body" | grep -q '"drained":true'; then
            streak=$(( streak + 1 ))
            [ "$streak" -ge 3 ] && return 0
        else
            streak=0
            echo "    queue not drained yet: ${body:-unreachable}"
        fi
        sleep 5
    done
    echo "  WARNING: queue did not drain within ${DRAIN_TIMEOUT}s - next run may be contaminated"
    return 1
}

echo "========================================================"
echo "  XAI Load Test Suite (load-generator EC2 edition)"
echo "  Host     : $TARGET_HOST"
echo "  Run time : $RUN_TIME per experiment"
echo "  Repeats  : $REPEATS per (architecture, concurrency) combo"
echo "  Total    : $total runs"
echo "  Output   : $RESULTS/"
echo "========================================================"

for i in "${!ARCHS[@]}"; do
    arch="${ARCHS[$i]}"
    file="${FILES[$i]}"

    echo ""
    echo "--- Architecture: $arch ---"

    for n in "${LEVELS[@]}"; do
        for rep in $(seq 1 "$REPEATS"); do
            run_count=$(( run_count + 1 ))
            base_name="locust_${arch}_${n}u_run${rep}"
            csv_name="$RESULTS/$base_name"

            echo ""
            echo "  [$run_count/$total]  $arch @ $n users (run $rep/$REPEATS) -> $STATS_DIR/${base_name}_stats.csv"

            locust -f "$file" \
                --host "$TARGET_HOST" \
                --users "$n" \
                --spawn-rate "$SPAWN_RATE" \
                --run-time "$RUN_TIME" \
                --csv "$csv_name" \
                --headless \
                --loglevel WARNING
            locust_exit=$?

            mv -f "${csv_name}_stats.csv"         "$STATS_DIR/${base_name}_stats.csv"           2>/dev/null
            mv -f "${csv_name}_stats_history.csv" "$HISTORY_DIR/${base_name}_stats_history.csv" 2>/dev/null
            mv -f "${csv_name}_failures.csv"      "$FAILURES_DIR/${base_name}_failures.csv"     2>/dev/null
            mv -f "${csv_name}_exceptions.csv"    "$EXCEPTIONS_DIR/${base_name}_exceptions.csv" 2>/dev/null

            if [ "$locust_exit" -ne 0 ]; then
                echo "  WARNING: locust reported failures for $arch @ $n users (normal under heavy load)"
            else
                echo "  Done."
            fi

            if [ "$run_count" -lt "$total" ]; then
                # Only asynch enqueues Celery work; baseline/synch have nothing to drain
                if [ "$arch" = "asynch" ]; then
                    echo "  Waiting for Celery Tier-2 queue to drain..."
                    wait_queue_drain || true
                fi
                echo "  Sleeping ${RECOVERY_SLEEP}s..."
                sleep "$RECOVERY_SLEEP"
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "  All $total experiments complete."
echo "  CSVs written to: $RESULTS/"
echo "  Copy back to your laptop, e.g. from the laptop:"
echo "    scp -i xai-key.pem -r ubuntu@<LOADGEN_PUBLIC_IP>:~/xai-throughput-optimizer/$RESULTS ."
echo "  Then: python analysis/mann_whitney.py $RESULTS"
echo "========================================================"
