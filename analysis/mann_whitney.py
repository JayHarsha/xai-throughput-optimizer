"""
Mann-Whitney U Test — XAI Pipeline Latency Comparison
NCI Dublin MSc AI Practicum — Dissertation Statistical Validation

Reads Locust *_stats.csv files from the results/ directory and compares p95
latency distributions across three architectures:
  - baseline  /predict/baseline  (pure inference, no XAI — control group)
  - synch     /predict/synch     (synchronous XAI, fully blocking)
  - asynch    /predict/asynch    (async Tier-1 perceived latency)

HOW TO PRODUCE THE INPUT FILES
-------------------------------
Run each Locust file once per concurrency level (one 60-second run
internally samples hundreds of requests, giving a stable p95 figure).

    .\scripts\benchmark_pipelines.ps1        # runs all 15 automatically

Or manually:
    locust -f tests/locust_baseline.py --host=http://localhost:8000 \\
           --users 50 --spawn-rate 5 --run-time 60s \\
           --csv=results/locust_baseline_50u_run1 --headless

Expected filename pattern:  locust_<arch>_<N>u_run<R>_stats.csv
Concurrency levels: 1, 5, 10, 25, 50  (one run each = 5 data points per architecture)
Effect sizes are large enough that n=5 gives p < 0.05 with high confidence.

RUNNING THE ANALYSIS
---------------------
    python analysis/mann_whitney.py
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

RESULTS_DIR = Path(__file__).parent.parent / "results"

ENDPOINT_MAP = {
    "baseline": "/predict/baseline",
    "synch":    "/predict/synch",
    "asynch":   "/predict/asynch",
}

# Locust stats CSV filenames must match: locust_<arch>_<N>u_run<R>_stats.csv
FILENAME_RE = re.compile(r"locust_(baseline|synch|asynch)_\d+u_run\d+_stats\.csv$")


def extract_p95(csv_path: Path, arch: str) -> float | None:
    """
    Parse one Locust stats CSV and return the p95 latency (ms).

    Locust CSV format:
      Type,  Name,               ..., 95%, ...
      POST,  /predict/baseline,  ..., 22,  ...   ← endpoint row  (we want this)
      "",    Aggregated,         ..., 22,  ...   ← summary row   (fallback)

    The two rows are identical when a locust file hits only one endpoint.
    """
    endpoint_path = ENDPOINT_MAP[arch]
    try:
        df = pd.read_csv(csv_path)
        # Primary: match by path in Name column + POST method
        row = df[(df["Name"] == endpoint_path) & (df["Type"].str.upper() == "POST")]
        if row.empty:
            # Fallback: Aggregated row (same numbers, different label)
            row = df[df["Name"] == "Aggregated"]
        if row.empty:
            print(f"  WARNING: no usable row found in {csv_path.name}")
            return None
        return float(row["95%"].iloc[0])
    except Exception as e:
        print(f"  ERROR reading {csv_path.name}: {e}")
        return None


def collect_p95(arch: str) -> list[float]:
    """Return all p95 values found across every matching CSV for this arch."""
    values = []
    files = sorted(RESULTS_DIR.glob(f"locust_{arch}_*_stats.csv"))
    matched = [f for f in files if FILENAME_RE.search(f.name)]
    for path in matched:
        p95 = extract_p95(path, arch)
        if p95 is not None:
            values.append(p95)
    return values


REPORT_PATH = RESULTS_DIR / "mann_whitney_report.txt"

# All output goes to console AND the report file simultaneously
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()


def descriptive(name: str, data: list[float]) -> None:
    arr = np.array(data)
    print(f"  {name}  (n={len(arr)} observations)")
    print(f"    median = {np.median(arr):.1f} ms")
    print(f"    mean   = {arr.mean():.1f} ms")
    print(f"    p95    = {np.percentile(arr, 95):.1f} ms")
    print(f"    max    = {arr.max():.1f} ms")


def mw_test(
    name_a: str, a: list[float],
    name_b: str, b: list[float],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    stat, p = mannwhitneyu(a, b, alternative=alternative)
    significant = p < 0.05
    verdict = "SIGNIFICANT  ✓" if significant else "not significant"
    print(f"\n  [{alternative}]  {name_a}  vs  {name_b}")
    print(f"  U = {stat:.1f}    p = {p:.4f}    {verdict}")
    print(f"  Median — {name_a}: {np.median(a):.1f} ms  |  {name_b}: {np.median(b):.1f} ms")
    return stat, p


if __name__ == "__main__":
    report_file = open(REPORT_PATH, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, report_file)

    print(f"Scanning results directory: {RESULTS_DIR}\n")

    baseline_p95 = collect_p95("baseline")
    sync_p95     = collect_p95("synch")
    async_p95    = collect_p95("asynch")

    missing = [
        name for name, data in [
            ("baseline", baseline_p95),
            ("synch",    sync_p95),
            ("asynch",   async_p95),
        ]
        if not data
    ]
    if missing:
        print("ERROR: No result CSV files found for:", missing)
        print()
        print("Run your load tests first, e.g.:")
        print("  locust -f tests/locust_baseline.py --host=http://localhost:8000 \\")
        print("         --users 50 --spawn-rate 5 --run-time 60s \\")
        print("         --csv=results/locust_baseline_50u_run1 --headless")
        print()
        print("Expected pattern: results/locust_<arch>_<N>u_run<R>_stats.csv")
        sys.exit(1)

    # ── Descriptive statistics ──────────────────────────────────────────────
    print("=" * 62)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 62)
    descriptive("No-XAI Baseline", baseline_p95)
    print()
    descriptive("Sync XAI",        sync_p95)
    print()
    descriptive("Async Pipeline",  async_p95)

    # ── Hypothesis tests ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("MANN-WHITNEY U TESTS  (non-parametric, α = 0.05)")
    print("=" * 62)

    print("\n  H1: XAI computation adds measurable latency overhead")
    mw_test("Sync XAI", sync_p95, "No-XAI Baseline", baseline_p95, alternative="greater")

    print("\n  H2: Async Tier-1 is faster than synchronous XAI")
    mw_test("Sync XAI", sync_p95, "Async Pipeline", async_p95, alternative="greater")

    print("\n  H3: Async Tier-1 perceived latency matches the no-XAI floor")
    mw_test("Async Pipeline", async_p95, "No-XAI Baseline", baseline_p95, alternative="two-sided")

    # ── Interpretation guide ────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("INTERPRETATION")
    print("=" * 62)
    print("  H1 significant  → XAI overhead is real and measurable")
    print("  H2 significant  → Async offloading eliminates that overhead for the user")
    print("  H3 NOT sig.     → Async Tier-1 latency is statistically the same as")
    print("                     pure inference — XAI is effectively free from the")
    print("                     user's perspective")
    print()

    sys.stdout = sys.__stdout__
    report_file.close()
    print(f"Report saved to: {REPORT_PATH}")
