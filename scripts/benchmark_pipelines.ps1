param(
    [string]$TargetHost = "http://localhost:8000",
    [string]$ResultsDir = "results"
)

$RESULTS        = $ResultsDir
$STATS_DIR      = "$RESULTS/stats"
$HISTORY_DIR    = "$RESULTS/history"
$FAILURES_DIR   = "$RESULTS/failures"
$EXCEPTIONS_DIR = "$RESULTS/exceptions"
$RUN_TIME       = "60s"
$SPAWN_RATE     = 5
$RECOVERY_SLEEP = 15
$REPEATS        = 3

New-Item -ItemType Directory -Force $RESULTS        | Out-Null
New-Item -ItemType Directory -Force $STATS_DIR      | Out-Null
New-Item -ItemType Directory -Force $HISTORY_DIR    | Out-Null
New-Item -ItemType Directory -Force $FAILURES_DIR   | Out-Null
New-Item -ItemType Directory -Force $EXCEPTIONS_DIR | Out-Null

$ARCHS  = @("baseline",                "synch",                "asynch")
$FILES  = @("tests/locust_baseline.py","tests/locust_synch.py","tests/locust_asynch.py")
$LEVELS = @(1, 5, 10, 25, 50)

$total     = $ARCHS.Count * $LEVELS.Count * $REPEATS
$run_count = 0

Write-Host "========================================================"
Write-Host "  XAI Load Test Suite"
Write-Host "  Host     : $TargetHost"
Write-Host "  Run time : $RUN_TIME per experiment"
Write-Host "  Repeats  : $REPEATS per (architecture, concurrency) combo"
Write-Host "  Total    : $total runs"
Write-Host "  Output   : $RESULTS/"
Write-Host "========================================================"

for ($i = 0; $i -lt $ARCHS.Count; $i++) {
    $arch = $ARCHS[$i]
    $file = $FILES[$i]

    Write-Host ""
    Write-Host "--- Architecture: $arch ---"

    foreach ($n in $LEVELS) {
        for ($rep = 1; $rep -le $REPEATS; $rep++) {
            $run_count++
            $base_name = "locust_${arch}_${n}u_run${rep}"
            $csv_name  = "$RESULTS/$base_name"

            Write-Host ""
            Write-Host "  [$run_count/$total]  $arch @ $n users (run $rep/$REPEATS) -> $STATS_DIR/${base_name}_stats.csv"

            $locustArgs = @(
                "-f", $file,
                "--host", $TargetHost,
                "--users", $n,
                "--spawn-rate", $SPAWN_RATE,
                "--run-time", $RUN_TIME,
                "--csv", $csv_name,
                "--headless",
                "--loglevel", "WARNING"
            )

            & locust @locustArgs

            # Route each Locust output file into its own folder by type
            # (stats/ is the only one analysis/mann_whitney.py reads; the rest are kept as raw evidence)
            Move-Item -Force "${csv_name}_stats.csv"         "$STATS_DIR/${base_name}_stats.csv"           -ErrorAction SilentlyContinue
            Move-Item -Force "${csv_name}_stats_history.csv" "$HISTORY_DIR/${base_name}_stats_history.csv" -ErrorAction SilentlyContinue
            Move-Item -Force "${csv_name}_failures.csv"      "$FAILURES_DIR/${base_name}_failures.csv"     -ErrorAction SilentlyContinue
            Move-Item -Force "${csv_name}_exceptions.csv"    "$EXCEPTIONS_DIR/${base_name}_exceptions.csv" -ErrorAction SilentlyContinue

            if ($LASTEXITCODE -ne 0) {
                Write-Host "  WARNING: locust reported failures for $arch @ $n users (normal under heavy load)"
            } else {
                Write-Host "  Done."
            }

            if ($run_count -lt $total) {
                Write-Host "  Sleeping ${RECOVERY_SLEEP}s..."
                Start-Sleep -Seconds $RECOVERY_SLEEP
            }
        }
    }
}

Write-Host ""
Write-Host "========================================================"
Write-Host "  All $total experiments complete."
Write-Host "  CSVs written to: $RESULTS/"
Write-Host "  Next: python analysis/mann_whitney.py"
Write-Host "========================================================"
