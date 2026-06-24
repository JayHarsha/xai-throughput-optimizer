param(
    [string]$TargetHost = "http://localhost:8000"
)

$RESULTS       = "results"
$RUN_TIME      = "60s"
$SPAWN_RATE    = 5
$RECOVERY_SLEEP = 15

New-Item -ItemType Directory -Force $RESULTS | Out-Null

$ARCHS  = @("baseline",                "synch",                "asynch")
$FILES  = @("tests/locust_baseline.py","tests/locust_synch.py","tests/locust_asynch.py")
$LEVELS = @(1, 5, 10, 25, 50)

$total     = $ARCHS.Count * $LEVELS.Count
$run_count = 0

Write-Host "========================================================"
Write-Host "  XAI Load Test Suite"
Write-Host "  Host     : $TargetHost"
Write-Host "  Run time : $RUN_TIME per experiment"
Write-Host "  Total    : $total runs"
Write-Host "  Output   : $RESULTS/"
Write-Host "========================================================"

for ($i = 0; $i -lt $ARCHS.Count; $i++) {
    $arch = $ARCHS[$i]
    $file = $FILES[$i]

    Write-Host ""
    Write-Host "--- Architecture: $arch ---"

    foreach ($n in $LEVELS) {
        $run_count++
        $csv_name = "$RESULTS/locust_${arch}_${n}u_run1"

        Write-Host ""
        Write-Host "  [$run_count/$total]  $arch @ $n users -> ${csv_name}_stats.csv"

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

        # Always remove junk files — Locust exits non-zero even on partial failures
        # so we cannot put this inside an else block
        Remove-Item -Force "${csv_name}_stats_history.csv" -ErrorAction SilentlyContinue
        Remove-Item -Force "${csv_name}_failures.csv"      -ErrorAction SilentlyContinue
        Remove-Item -Force "${csv_name}_exceptions.csv"    -ErrorAction SilentlyContinue

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

Write-Host ""
Write-Host "========================================================"
Write-Host "  All $total experiments complete."
Write-Host "  CSVs written to: $RESULTS/"
Write-Host "  Next: python analysis/mann_whitney.py"
Write-Host "========================================================"
