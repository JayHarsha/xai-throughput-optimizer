"""
Statistical Analysis — XAI Pipeline Latency Comparison
Non-parametric Mann-Whitney U Tests (alpha = 0.05)

Research Narrative
──────────────────
Three hypotheses form a sequential chain of evidence:

  Step 1 — Establish the problem
  H1  Does synchronous XAI execution create a measurable latency
      overhead compared to pure ML inference?
      [one-sided: Sync > Baseline]
      → If confirmed: synchronous XAI is architecturally unsafe at scale.

  Step 2 — Validate the proposed solution
  H2  Does the async pipeline significantly reduce tail latency
      compared to synchronous XAI?
      [one-sided: Sync > Async]
      → If confirmed: async decoupling is a statistically validated fix.

  Step 3 — Test the ideal outcome
  H3  Does the async pipeline match the no-XAI latency floor,
      making XAI computationally "free" from the user's perspective?
      [two-sided: Async ≈ Baseline]
      → If confirmed (p > 0.05): perfect. If rejected: Tier-1 SHAP
        still adds overhead under load — a scope limitation for future work.

Input files:  results/locust_<arch>_<N>u_run<R>_stats.csv
              arch in {baseline, synch, asynch},  N in {1, 5, 10, 25, 50}

Outputs:      results/mann_whitney_report.txt
              results/summary_table.csv
              results/plots/  (7 figures)
"""

import math
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu

# ── Paths ────────────────────────────────────────────────────────────────────
# Optional CLI arg picks which results folder to analyze, e.g.:
#   python analysis/mann_whitney.py            -> results/       (local runs)
#   python analysis/mann_whitney.py aws_results -> aws_results/  (AWS runs)
RESULTS_DIR_NAME = sys.argv[1] if len(sys.argv) > 1 else "results"
RESULTS_DIR      = Path(__file__).parent.parent / RESULTS_DIR_NAME
STATS_DIR    = RESULTS_DIR / "stats"      # only *_stats.csv is read for the Mann-Whitney report
FAILURES_DIR = RESULTS_DIR / "failures"   # *_failures.csv — read only for the separate failure breakdown
PLOTS_DIR    = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

REPORT_PATH            = RESULTS_DIR / "mann_whitney_report.txt"
SUMMARY_PATH           = RESULTS_DIR / "summary_table.csv"
FAILURE_BREAKDOWN_PATH = RESULTS_DIR / "failure_breakdown.txt"

# ── Constants ────────────────────────────────────────────────────────────────
CONCURRENCY_LEVELS = [1, 5, 10, 25, 50]

ENDPOINT_MAP = {
    "baseline": "/predict/baseline",
    "synch":    "/predict/synch",
    "asynch":   "/predict/asynch",
}

ARCH_LABELS = {
    "baseline": "No XAI",
    "synch":    "Synch XAI",
    "asynch":   "Asynch XAI",
}

COLORS = {
    "baseline": "#22c55e",
    "synch":    "#ef4444",
    "asynch":   "#3b82f6",
}

ARCHS = ["baseline", "synch", "asynch"]
FILENAME_RE          = re.compile(r"locust_(baseline|synch|asynch)_(\d+)u_run\d+_stats\.csv$")
FILENAME_RE_FAILURES = re.compile(r"locust_(baseline|synch|asynch)_(\d+)u_run\d+_failures\.csv$")


# ── Data extraction ──────────────────────────────────────────────────────────
def parse_csv(csv_path: Path, arch: str) -> dict | None:
    endpoint = ENDPOINT_MAP[arch]
    try:
        df  = pd.read_csv(csv_path)
        row = df[(df["Name"] == endpoint) & (df["Type"].str.upper() == "POST")]
        if row.empty:
            row = df[df["Name"] == "Aggregated"]
        if row.empty:
            return None
        r      = row.iloc[0]
        total  = int(r["Request Count"])
        fails  = int(r["Failure Count"])
        return {
            "p50":           float(r["50%"]),
            "p95":           float(r["95%"]),
            "p99":           float(r["99%"]),
            "req_s":         float(r["Requests/s"]),
            "failure_rate":  round(fails / total * 100, 1) if total > 0 else 0.0,
            "request_count": total,
            "failure_count": fails,
            "min_ms":        float(r["Min Response Time"]),
            "max_ms":        float(r["Max Response Time"]),
            "avg_ms":        float(r["Average Response Time"]),
        }
    except Exception as e:
        print(f"  ERROR reading {csv_path.name}: {e}")
        return None


def collect_all(arch: str) -> dict[int, list[dict]]:
    """Returns {concurrency_level: [run1_dict, run2_dict, ...]} — supports repeat runs per level."""
    result: dict[int, list[dict]] = {}
    for path in sorted(STATS_DIR.glob(f"locust_{arch}_*_stats.csv")):
        m = FILENAME_RE.search(path.name)
        if not m:
            continue
        n    = int(m.group(2))
        data = parse_csv(path, arch)
        if data:
            result.setdefault(n, []).append(data)
    return result


def average_runs(runs: list[dict]) -> dict:
    """Aggregate repeat runs at one concurrency level into a single summary point."""
    total = sum(r["request_count"] for r in runs)
    fails = sum(r["failure_count"] for r in runs)
    return {
        "p50":           float(np.mean([r["p50"] for r in runs])),
        "p95":           float(np.mean([r["p95"] for r in runs])),
        "p95_std":       float(np.std([r["p95"] for r in runs])),
        "p99":           float(np.mean([r["p99"] for r in runs])),
        "req_s":         float(np.mean([r["req_s"] for r in runs])),
        "failure_rate":  round(fails / total * 100, 1) if total > 0 else 0.0,
        "request_count": total,
        "failure_count": fails,
        "n_runs":        len(runs),
    }


# ── Failure reason breakdown (separate output, not part of the Mann-Whitney report) ───────────
def collect_failure_reasons(arch: str) -> dict[int, dict[str, int]]:
    """Returns {concurrency_level: {error_type: total_occurrences_across_repeat_runs}}."""
    result: dict[int, dict[str, int]] = {}
    for path in sorted(FAILURES_DIR.glob(f"locust_{arch}_*_failures.csv")):
        m = FILENAME_RE_FAILURES.search(path.name)
        if not m:
            continue
        n = int(m.group(2))
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        level_counts = result.setdefault(n, {})
        for _, row in df.iterrows():
            error       = str(row.get("Error", "Unknown"))
            occurrences = int(row.get("Occurrences", 0))
            level_counts[error] = level_counts.get(error, 0) + occurrences
    return result


def write_failure_breakdown(failure_data: dict[str, dict[int, dict[str, int]]]) -> None:
    with open(FAILURE_BREAKDOWN_PATH, "w", encoding="utf-8") as f:
        print("=" * 68, file=f)
        print("  Failure Reason Breakdown", file=f)
        print("  Source: results/failures/*_failures.csv, summed across repeat runs", file=f)
        print("  (Not part of the Mann-Whitney hypothesis tests — diagnostic only)", file=f)
        print("=" * 68, file=f)

        for arch in ARCHS:
            print(f"\n  {ARCH_LABELS[arch]}", file=f)
            print("  " + "-" * 60, file=f)
            levels = failure_data.get(arch, {})
            if not levels:
                print("    No failures recorded.", file=f)
                continue
            for n in CONCURRENCY_LEVELS:
                errors = levels.get(n)
                if not errors:
                    continue
                total = sum(errors.values())
                print(f"    {n} users  ({total} total failures)", file=f)
                for error, count in sorted(errors.items(), key=lambda x: -x[1]):
                    pct = count / total * 100
                    print(f"      {count:>5}  ({pct:5.1f}%)  {error}", file=f)
        print("", file=f)
    print(f"Failure breakdown saved  -> {FAILURE_BREAKDOWN_PATH}")


# ── Tier-2 completion latency (Async only) ──────────────────────────────────
# Measures the real end-to-end time until the background SHAP-interactions +
# counterfactual analysis actually finishes, via locust_asynch.py's "tier2_completion"
# custom metric — distinct from the fast Tier-1 "/predict/asynch" acknowledgment above.
def parse_tier2_csv(csv_path: Path) -> dict | None:
    try:
        df  = pd.read_csv(csv_path)
        row = df[(df["Name"] == "tier2_completion") & (df["Type"].str.upper() == "POLL")]
        if row.empty:
            return None
        r     = row.iloc[0]
        total = int(r["Request Count"])
        fails = int(r["Failure Count"])
        return {
            "p50":           float(r["50%"]),
            "p95":           float(r["95%"]),
            "p99":           float(r["99%"]),
            "avg_ms":        float(r["Average Response Time"]),
            "request_count": total,
            "failure_count": fails,
        }
    except Exception as e:
        print(f"  ERROR reading {csv_path.name} for tier2_completion: {e}")
        return None


def collect_tier2_completion() -> dict[int, list[dict]]:
    """Returns {concurrency_level: [run1_dict, run2_dict, ...]} for Async's Tier-2 metric.
    Gracefully returns {} if the CSVs predate the polling instrumentation."""
    result: dict[int, list[dict]] = {}
    for path in sorted(STATS_DIR.glob("locust_asynch_*_stats.csv")):
        m = FILENAME_RE.search(path.name)
        if not m:
            continue
        n    = int(m.group(2))
        data = parse_tier2_csv(path)
        if data:
            result.setdefault(n, []).append(data)
    return result


def average_tier2_runs(runs: list[dict]) -> dict:
    total = sum(r["request_count"] for r in runs)
    fails = sum(r["failure_count"] for r in runs)
    return {
        "p50":           float(np.mean([r["p50"] for r in runs])),
        "p95":           float(np.mean([r["p95"] for r in runs])),
        "p99":           float(np.mean([r["p99"] for r in runs])),
        "avg_ms":        float(np.mean([r["avg_ms"] for r in runs])),
        "request_count": total,
        "failure_count": fails,
        "n_runs":        len(runs),
    }


# ── Statistics helpers ───────────────────────────────────────────────────────
def rank_biserial(u: float, n1: int, n2: int, alternative: str = "greater") -> float:
    """
    Rank-biserial correlation r, range [-1, +1].
    Directional: r = 2U/(n1*n2) - 1  (+1 = perfect separation in claimed direction)
    Two-sided:   r = |2U/(n1*n2) - 1|  (magnitude)
    """
    r = (2 * u) / (n1 * n2) - 1
    return round(abs(r) if alternative == "two-sided" else r, 3)


def effect_label(r: float) -> str:
    a = abs(r)
    if a >= 0.5: return "large"
    if a >= 0.3: return "medium"
    return "small"


# ── Output helpers ───────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def section(title: str) -> None:
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def subsection(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(0, 58 - len(title))}")


# ── Mann-Whitney test runner ─────────────────────────────────────────────────
def run_mw(
    step: str,
    name_a: str, a: list[float],
    name_b: str, b: list[float],
    hypothesis: str,
    plain_english: str,
    alternative: str,
) -> dict:
    stat, p  = mannwhitneyu(a, b, alternative=alternative)
    n1, n2   = len(a), len(b)
    r        = rank_biserial(stat, n1, n2, alternative)
    sig      = p < 0.05
    # NOTE: a non-significant two-sided test is NOT evidence of equivalence
    # (absence of evidence fallacy) — the formal equivalence claim is made by
    # the TOST procedure printed directly below this test.
    if alternative == "two-sided":
        verdict = ("NO significant difference detected  (descriptive only — see TOST below for the formal equivalence test)"
                   if not sig else
                   "Distributions differ  (p < 0.05 — see TOST below for the bounded-equivalence margin)")
    else:
        verdict = "SUPPORTED  ✓" if sig else "NOT SUPPORTED  ✗"

    print(f"\n  {step}: {hypothesis}")
    print(f"  Research question : {plain_english}")
    print(f"  Test              : Mann-Whitney U [{alternative}],  {name_a}  vs  {name_b}")
    print(f"  U statistic       : {stat:.1f}")
    print(f"  p-value           : {p:.4f}   {'(< 0.05 — significant)' if sig else '(> 0.05 — not significant)'}")
    print(f"  Effect size       : r = {r}  ({effect_label(r)} effect)")
    print(f"  Median {name_a:<18}: {np.median(a):>8,.0f} ms")
    print(f"  Median {name_b:<18}: {np.median(b):>8,.0f} ms")
    print(f"  Verdict           : {verdict}")

    return {"stat": stat, "p": p, "r": r, "sig": sig, "alternative": alternative}


# ── Equivalence testing (TOST) ───────────────────────────────────────────────
TOST_MARGINS = [1.1, 1.25, 1.5, 2.0, 3.0]   # multiplicative margins θ
TOST_PRIMARY_MARGIN = 1.5                    # pre-registered headline margin

def run_tost(a: list[float], b: list[float], theta: float) -> dict:
    """
    Two One-Sided Tests for equivalence on log-latencies with a multiplicative
    margin θ: H0 is 'a differs from b by more than a factor of θ'. Equivalence
    is claimed only if BOTH one-sided Mann-Whitney tests reject at α = 0.05,
    i.e. p_tost = max(p_upper, p_lower) < 0.05. This is the statistically
    correct way to claim 'async ≈ baseline' — a non-significant two-sided
    p-value alone would be the absence-of-evidence fallacy.
    """
    la = np.log(np.maximum(np.asarray(a, dtype=float), 0.1))
    lb = np.log(np.maximum(np.asarray(b, dtype=float), 0.1))
    lt = np.log(theta)
    # a is within θx above b:  reject H0 that (a / θ) >= b
    _, p_upper = mannwhitneyu(la - lt, lb, alternative="less")
    # a is within θx below b:  reject H0 that (a * θ) <= b
    _, p_lower = mannwhitneyu(la + lt, lb, alternative="greater")
    p_tost = max(p_upper, p_lower)
    return {"theta": theta, "p_upper": p_upper, "p_lower": p_lower,
            "p": p_tost, "equivalent": p_tost < 0.05}


def run_tost_sweep(step: str, name_a: str, a: list[float],
                   name_b: str, b: list[float]) -> dict:
    """Runs TOST across TOST_MARGINS, prints the sweep, returns the primary-margin
    result plus θ* (the smallest margin at which equivalence holds, if any)."""
    print(f"\n  {step}: TOST equivalence test  —  {name_a} vs {name_b}")
    print(f"  Margins are multiplicative: θ = 1.5 means 'within 1.5x of each other'")
    print(f"  {'θ':>6}  {'p_upper':>9}  {'p_lower':>9}  {'p_TOST':>9}  verdict")
    results, theta_star = [], None
    for theta in TOST_MARGINS:
        res = run_tost(a, b, theta)
        results.append(res)
        if res["equivalent"] and theta_star is None:
            theta_star = theta
        print(f"  {theta:>5.2f}x  {res['p_upper']:>9.4f}  {res['p_lower']:>9.4f}  "
              f"{res['p']:>9.4f}  {'EQUIVALENT within this margin' if res['equivalent'] else 'not shown equivalent'}")
    primary = next(r for r in results if r["theta"] == TOST_PRIMARY_MARGIN)
    if theta_star is not None:
        print(f"  θ* (smallest margin with demonstrated equivalence): {theta_star}x")
    else:
        print(f"  θ*: equivalence not demonstrated at any tested margin (max {TOST_MARGINS[-1]}x)")
    print(f"  Primary verdict (pre-registered θ = {TOST_PRIMARY_MARGIN}x): "
          f"{'EQUIVALENT' if primary['equivalent'] else 'NOT demonstrated equivalent'}  (p = {primary['p']:.4f})")
    return {"primary": primary, "theta_star": theta_star, "sweep": results}


# ── Per-concurrency-level tests with Holm-Bonferroni correction ──────────────
def holm_correction(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (controls family-wise error rate)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted.tolist()


def per_level_tests(raw_data: dict) -> None:
    """
    Repeats H1/H2 within each concurrency level separately (repeat-run p95s,
    n = repeats per group), avoiding the criticism that pooling across levels
    mixes non-exchangeable operating points. Holm correction applied across
    the five levels per hypothesis.
    """
    subsection("Per-concurrency-level tests (Holm-corrected across levels)")
    n_reps = min(
        (len(raw_data[arch][n]) for arch in ARCHS for n in raw_data[arch]),
        default=0,
    )
    if n_reps < 4:
        print(f"""
  NOTE: with only {n_reps} repeats per cell the smallest achievable one-sided
  exact Mann-Whitney p-value is {1/math.comb(2*n_reps, n_reps):.3f} — per-level significance is
  underpowered and shown descriptively. Run the suite with REPEATS >= 5
  (now the default in scripts/benchmark_pipelines.*) for confirmatory
  per-level inference; the pooled tests above remain the primary analysis.""")

    for hyp, arch_a, arch_b, label in [
        ("H1", "synch", "baseline", "Synch > No-XAI"),
        ("H2", "synch", "asynch",   "Synch > Asynch"),
    ]:
        rows, pvals = [], []
        for n in CONCURRENCY_LEVELS:
            a = [r["p95"] for r in raw_data[arch_a].get(n, [])]
            b = [r["p95"] for r in raw_data[arch_b].get(n, [])]
            if not a or not b:
                continue
            stat, p = mannwhitneyu(a, b, alternative="greater")
            rows.append((n, np.median(a), np.median(b), p))
            pvals.append(p)
        adj = holm_correction(pvals)
        print(f"\n  {hyp} per level: {label}  (one-sided MW on repeat-run p95s)")
        print(f"  {'users':>6}  {'median A (ms)':>14}  {'median B (ms)':>14}  {'p':>8}  {'p (Holm)':>9}  sig")
        for (n, ma, mb, p), pa in zip(rows, adj):
            print(f"  {n:>6}  {ma:>14,.0f}  {mb:>14,.0f}  {p:>8.4f}  {pa:>9.4f}  {'*' if pa < 0.05 else ''}")

    # H3 per level: descriptive ratio + TOST at the primary margin
    print(f"\n  H3 per level: Asynch vs No-XAI  (p95 ratio + TOST at θ = {TOST_PRIMARY_MARGIN}x)")
    print(f"  {'users':>6}  {'async p95':>10}  {'no-XAI p95':>11}  {'ratio':>6}  {'p_TOST':>8}  verdict")
    for n in CONCURRENCY_LEVELS:
        a = [r["p95"] for r in raw_data["asynch"].get(n, [])]
        b = [r["p95"] for r in raw_data["baseline"].get(n, [])]
        if not a or not b:
            continue
        res = run_tost(a, b, TOST_PRIMARY_MARGIN)
        ratio = np.median(a) / max(np.median(b), 0.1)
        print(f"  {n:>6}  {np.median(a):>10,.0f}  {np.median(b):>11,.0f}  {ratio:>5.2f}x  {res['p']:>8.4f}  "
              f"{'equivalent' if res['equivalent'] else 'not shown equivalent'}")


# ── Summary table ────────────────────────────────────────────────────────────
def build_summary(all_data: dict) -> pd.DataFrame:
    rows = []
    for arch in ARCHS:
        for n in CONCURRENCY_LEVELS:
            d = all_data[arch].get(n)
            if d is None:
                continue
            rows.append({
                "Architecture":      ARCH_LABELS[arch],
                "Concurrent Users":  n,
                "p50 (ms)":          int(d["p50"]),
                "p95 (ms)":          int(d["p95"]),
                "p99 (ms)":          int(d["p99"]),
                "Throughput (req/s)": round(d["req_s"], 2),
                "Failure Rate (%)":   d["failure_rate"],
                "Request Count":      d["request_count"],
                "Repeats":            d["n_runs"],
            })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)


def _series(all_data, metric):
    return {
        arch: (
            [n for n in CONCURRENCY_LEVELS if n in all_data[arch]],
            [all_data[arch][n][metric] for n in CONCURRENCY_LEVELS if n in all_data[arch]]
        )
        for arch in ARCHS
    }


def save(fig, name):
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def plot_p95_latency(all_data):
    """Figure 1 — p95 tail latency vs concurrency (log scale). Hero figure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for arch in ARCHS:
        xs = [n for n in CONCURRENCY_LEVELS if n in all_data[arch]]
        ys = [all_data[arch][n]["p95"] for n in xs]
        errs = [all_data[arch][n].get("p95_std", 0) for n in xs]
        ax.errorbar(xs, ys, yerr=errs, marker="o", linewidth=2.5, markersize=8,
                    capsize=4, color=COLORS[arch], label=ARCH_LABELS[arch])
        for x, y in zip(xs, ys):
            ax.annotate(f"{int(y):,} ms", xy=(x, y), xytext=(5, 7),
                        textcoords="offset points", fontsize=8, color=COLORS[arch])
    ax.axhspan(0, 1000, alpha=0.04, color="green", label="< 1,000 ms zone")
    ax.axhline(1000, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(50.5, 1050, "1,000 ms", fontsize=8, color="gray", va="bottom")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,} ms"))
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.set_xlabel("Concurrent Users", fontsize=12)
    ax.set_ylabel("p95 Tail Latency (log scale)", fontsize=12)
    ax.set_title("p95 Tail Latency vs Concurrent Users  (error bars = std across repeat runs)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    save(fig, "fig1_p95_latency.png")


def plot_percentiles(all_data):
    """Figure 2 — p50 / p95 / p99 per architecture, three subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, (metric, title) in zip(axes, [("p50", "p50 Median"), ("p95", "p95 Tail"), ("p99", "p99 Extreme Tail")]):
        for arch in ARCHS:
            xs, ys = _series(all_data, metric)[arch]
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=6,
                    color=COLORS[arch], label=ARCH_LABELS[arch])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xticks(CONCURRENCY_LEVELS)
        ax.set_xlabel("Concurrent Users")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("Latency Percentile Comparison Across All Architectures", fontweight="bold")
    save(fig, "fig2_percentile_comparison.png")


def plot_throughput(all_data):
    """Figure 3 — throughput (req/s) vs concurrency."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for arch in ARCHS:
        xs, ys = _series(all_data, "req_s")[arch]
        ax.plot(xs, ys, marker="s", linewidth=2.2, markersize=7,
                color=COLORS[arch], label=ARCH_LABELS[arch])
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.set_xlabel("Concurrent Users")
    ax.set_ylabel("Throughput (Requests / Second)")
    ax.set_title("System Throughput vs Concurrent Users", fontweight="bold")
    ax.legend()
    save(fig, "fig3_throughput.png")


def plot_failure_rate(all_data):
    """Figure 4 — failure rate (%) vs concurrency."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for arch in ARCHS:
        xs, ys = _series(all_data, "failure_rate")[arch]
        ax.plot(xs, ys, marker="^", linewidth=2.2, markersize=7,
                color=COLORS[arch], label=ARCH_LABELS[arch])
        for x, y in zip(xs, ys):
            if y > 0:
                ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(4, 5),
                            textcoords="offset points", fontsize=8, color=COLORS[arch])
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.set_ylim(-3, 90)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Concurrent Users")
    ax.set_ylabel("Request Failure Rate (%)")
    ax.set_title("Request Failure Rate vs Concurrent Users", fontweight="bold")
    ax.legend()
    save(fig, "fig4_failure_rate.png")


def plot_boxplots(p95_samples: dict):
    """Figure 5 — box plots of p95 distributions (Mann-Whitney U visualisation)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data   = [p95_samples[a] for a in ARCHS]
    labels = [ARCH_LABELS[a] for a in ARCHS]
    colors = [COLORS[a] for a in ARCHS]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data, colors), start=1):
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter([i + j for j in jitter], vals, color=color, zorder=5,
                   s=55, edgecolors="white", linewidths=0.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticklabels(labels)
    ax.set_ylabel("p95 Tail Latency (ms)  —  log scale")
    ax.set_title(f"p95 Latency Distribution Across Concurrency Levels\n(Each point = one repeat run, {len(data[0])} total per architecture)", fontweight="bold")
    save(fig, "fig5_boxplots.png")


def plot_p95_heatmap(all_data):
    """
    Figure 6 — heatmap: architecture x concurrency, coloured by p95 latency (seconds).
    Immediately shows which cells are dangerous (dark red) vs safe (dark green).
    """
    matrix = []
    for arch in ARCHS:
        row = [all_data[arch].get(n, {}).get("p95", np.nan) / 1000
               for n in CONCURRENCY_LEVELS]
        matrix.append(row)
    df = pd.DataFrame(matrix,
                      index=[ARCH_LABELS[a] for a in ARCHS],
                      columns=[f"{n} users" for n in CONCURRENCY_LEVELS])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax,
                annot_kws={"size": 11, "weight": "bold"},
                cbar_kws={"label": "p95 Latency (seconds)", "shrink": 0.8})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_title("p95 Tail Latency Heatmap  (values in seconds)\nArchitecture × Concurrent Users",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Concurrent Users", fontsize=11, labelpad=8)
    ax.set_ylabel("")
    save(fig, "fig6_p95_heatmap.png")


def plot_speedup_ratio(all_data):
    """
    Figure 7 — grouped bar chart (linear scale) showing actual p95 latency in seconds
    for Synch XAI vs Asynch XAI. Speedup ratio annotated above each Asynch bar.
    Advantage peaks at 10 users (16.7x) then narrows as Tier-1 SHAP saturates workers.
    """
    xs, ratios, synch_s_vals, asynch_s_vals = [], [], [], []
    for n in CONCURRENCY_LEVELS:
        s = all_data["synch"].get(n, {}).get("p95")
        a = all_data["asynch"].get(n, {}).get("p95")
        if s and a and a > 0:
            xs.append(n)
            ratios.append(round(s / a, 1))
            synch_s_vals.append(s / 1000)
            asynch_s_vals.append(a / 1000)

    fig, ax = plt.subplots(figsize=(10, 6))
    x     = np.arange(len(xs))
    width = 0.35
    y_max = max(synch_s_vals) * 1.18

    bars_s = ax.bar(x - width / 2, synch_s_vals, width,
                    color=COLORS["synch"], alpha=0.85, label="Synch XAI",
                    edgecolor="white", linewidth=0.5)
    bars_a = ax.bar(x + width / 2, asynch_s_vals, width,
                    color=COLORS["asynch"], alpha=0.85, label="Asynch XAI",
                    edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars_s, synch_s_vals):
        lbl = f"{val:.1f}s" if val >= 1 else f"{val:.2f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, val + y_max * 0.01,
                lbl, ha="center", va="bottom", fontsize=8.5,
                color=COLORS["synch"], fontweight="bold")

    for bar, val in zip(bars_a, asynch_s_vals):
        lbl = f"{val:.2f}s" if val < 1 else f"{val:.1f}s"
        ax.text(bar.get_x() + bar.get_width() / 2, val + y_max * 0.01,
                lbl, ha="center", va="bottom", fontsize=8.5,
                color=COLORS["asynch"], fontweight="bold")

    for i, (ratio, a_val) in enumerate(zip(ratios, asynch_s_vals)):
        ax.text(x[i] + width / 2, a_val + y_max * 0.09,
                f"{ratio:.1f}×", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#333",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          alpha=0.92, edgecolor="#ccc", linewidth=0.6))

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in xs])
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Concurrent Users", fontsize=12)
    ax.set_ylabel("p95 Latency (seconds)", fontsize=12)
    ax.set_title(
        "Benefit of Asynchronous Decoupling\n"
        "p95 Latency: Synch XAI vs Asynch XAI  (×N = speedup factor)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    save(fig, "fig7_async_speedup.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    report_file = open(REPORT_PATH, "w", encoding="utf-8")
    sys.stdout  = _Tee(sys.__stdout__, report_file)

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 68)
    print("  XAI Pipeline Latency — Statistical Analysis Report")
    print("  Method: Mann-Whitney U Test (non-parametric, alpha = 0.05)")
    print("  Three architectures x Five concurrency levels (1,5,10,25,50 users)")
    print("=" * 68)

    # ── Load data ─────────────────────────────────────────────────────────────
    raw_data = {arch: collect_all(arch) for arch in ARCHS}
    missing  = [a for a in ARCHS if not raw_data[a]]
    if missing:
        print("ERROR: No result CSV files found for:", missing)
        sys.exit(1)

    # all_data: one averaged point per (arch, concurrency) — feeds the summary table and trend plots
    all_data = {
        arch: {n: average_runs(runs) for n, runs in raw_data[arch].items()}
        for arch in ARCHS
    }

    # p95_samples: every individual repeat run's p95 value, flattened across concurrency levels —
    # feeds the Mann-Whitney tests, so statistical power scales with REPEATS, not just concurrency levels
    p95_samples = {
        arch: [run["p95"] for n in CONCURRENCY_LEVELS if n in raw_data[arch] for run in raw_data[arch][n]]
        for arch in ARCHS
    }

    # ── Summary table ─────────────────────────────────────────────────────────
    section("FULL RESULTS TABLE")
    df = build_summary(all_data)
    print(df.to_string(index=False))
    df.to_csv(SUMMARY_PATH, index=False)
    print(f"\n  Saved to: {SUMMARY_PATH.name}")

    # ── Descriptive statistics ────────────────────────────────────────────────
    section("DESCRIPTIVE STATISTICS  —  p95 Tail Latency Distribution")
    n_each = len(p95_samples[ARCHS[0]])
    print(f"  (Each architecture has {n_each} observations: 5 concurrency levels x repeat runs)\n")
    for arch in ARCHS:
        a = np.array(p95_samples[arch])
        print(f"  {ARCH_LABELS[arch]}  ({len(a)} runs)")
        print(f"    Min            : {a.min():>8,.0f} ms")
        print(f"    Median         : {np.median(a):>8,.0f} ms")
        print(f"    Mean           : {a.mean():>8,.0f} ms")
        print(f"    Std Dev        : {a.std():>8,.0f} ms")
        print(f"    Max            : {a.max():>8,.0f} ms")
        print()

    # ── Hypothesis tests ──────────────────────────────────────────────────────
    section("HYPOTHESIS TESTS")
    print("""
  Research Narrative
  ──────────────────
  The three tests form a sequential chain of evidence designed to answer
  the central research question: Can asynchronous decoupling make XAI
  computationally transparent to the user under concurrent load?

  Step 1 establishes that the problem is real.
  Step 2 validates that the proposed solution addresses it.
  Step 3 tests whether the solution achieves the ideal outcome.
""")

    h1 = run_mw(
        step          = "Step 1  (Establish the problem)",
        name_a        = "Synch XAI",  a = p95_samples["synch"],
        name_b        = "No XAI",      b = p95_samples["baseline"],
        hypothesis    = "Synchronous XAI inflates tail latency beyond the no-XAI baseline",
        plain_english = "Is the latency problem caused by XAI, or by the API infrastructure?",
        alternative   = "greater",
    )

    h2 = run_mw(
        step          = "Step 2  (Validate the solution)",
        name_a        = "Synch XAI",  a = p95_samples["synch"],
        name_b        = "Asynch XAI",  b = p95_samples["asynch"],
        hypothesis    = "Async pipeline significantly reduces tail latency vs synchronous XAI",
        plain_english = "Does moving XAI off the request path actually fix the problem?",
        alternative   = "greater",
    )

    h3 = run_mw(
        step          = "Step 3  (Test the ideal outcome)",
        name_a        = "Asynch XAI",  a = p95_samples["asynch"],
        name_b        = "No XAI",      b = p95_samples["baseline"],
        hypothesis    = "Async Tier-1 perceived latency is statistically equivalent to no-XAI baseline",
        plain_english = "Is async so fast that XAI is effectively free from the user's perspective?",
        alternative   = "two-sided",
    )

    h3_tost = run_tost_sweep(
        step   = "Step 3 (formal)",
        name_a = "Asynch XAI", a = p95_samples["asynch"],
        name_b = "No XAI",     b = p95_samples["baseline"],
    )

    # ── Per-level analysis ────────────────────────────────────────────────────
    section("PER-CONCURRENCY-LEVEL ANALYSIS")
    per_level_tests(raw_data)

    # ── Interpretation ────────────────────────────────────────────────────────
    section("INTERPRETATION")

    print(f"""
  Step 1 — Problem established:
  {"CONFIRMED" if h1["sig"] else "NOT CONFIRMED"}
  Synch XAI adds {np.median(p95_samples["synch"])/np.median(p95_samples["baseline"]):.0f}x median
  latency overhead vs No XAI. The bottleneck is XAI computation,
  not API infrastructure. p = {h1["p"]:.4f}, r = {h1["r"]} ({effect_label(h1["r"])} effect).

  Step 2 — Solution validated:
  {"CONFIRMED" if h2["sig"] else "NOT CONFIRMED"}
  Asynch XAI reduces median p95 from {np.median(p95_samples["synch"]):,.0f} ms (Synch XAI) to
  {np.median(p95_samples["asynch"]):,.0f} ms — a {(np.median(p95_samples["synch"])-np.median(p95_samples["asynch"]))/np.median(p95_samples["synch"])*100:.0f}% reduction.
  Asynchronous decoupling via Redis/Celery is a statistically validated
  architectural fix. p = {h2["p"]:.4f}, r = {h2["r"]} ({effect_label(h2["r"])} effect).

  Step 3 — Ideal outcome (TOST equivalence, pre-registered margin {TOST_PRIMARY_MARGIN}x):
  {"ACHIEVED" if h3_tost["primary"]["equivalent"] else "NOT FULLY ACHIEVED (scope limitation)"}
  {f"Async Tier-1 p95 latency is formally equivalent to the no-XAI floor within a factor of {TOST_PRIMARY_MARGIN} (TOST p = {h3_tost['primary']['p']:.4f}). XAI is effectively free to the user." if h3_tost["primary"]["equivalent"] else
  (f"Equivalence to the no-XAI floor holds only within a factor of {h3_tost['theta_star']}x (TOST). " if h3_tost["theta_star"] is not None else
   f"Equivalence to the no-XAI floor could not be demonstrated at any tested margin up to {TOST_MARGINS[-1]}x. ")
  + f"Asynch XAI still reduces tail latency by ~{(np.median(p95_samples['synch'])-np.median(p95_samples['asynch']))/np.median(p95_samples['synch'])*100:.0f}% vs Synch XAI (Step 2 confirmed). Two-sided MW p = {h3['p']:.4f} (descriptive). Future work: precompute or cache Tier-1 explanations to close the residual gap."}
""")

    # ── Key quantitative findings ─────────────────────────────────────────────
    section("KEY QUANTITATIVE FINDINGS AT PEAK LOAD  (50 concurrent users)")

    b50 = all_data["baseline"].get(50, {})
    s50 = all_data["synch"].get(50, {})
    a50 = all_data["asynch"].get(50, {})

    print(f"""
  Tail latency (p95):
    No XAI     : {b50.get('p95', 0):>8,.0f} ms
    Synch XAI  : {s50.get('p95', 0):>8,.0f} ms  ({s50.get('p95',0)/max(b50.get('p95',1),1):.0f}x slower than No XAI)
    Asynch XAI : {a50.get('p95', 0):>8,.0f} ms  ({(s50.get('p95',0)-a50.get('p95',0))/max(s50.get('p95',1),1)*100:.0f}% faster than Synch XAI)

  Failure rate:
    No XAI     : {b50.get('failure_rate', 0):>6.1f}%
    Synch XAI  : {s50.get('failure_rate', 0):>6.1f}%   (system effectively broken under load)
    Asynch XAI : {a50.get('failure_rate', 0):>6.1f}%

  Throughput (req/s):
    No XAI     : {b50.get('req_s', 0):>6.2f} req/s
    Synch XAI  : {s50.get('req_s', 0):>6.2f} req/s
    Asynch XAI : {a50.get('req_s', 0):>6.2f} req/s
""")

    # ── Tier-2 completion latency (Async only) ──────────────────────────────────
    tier2_raw = collect_tier2_completion()
    tier2_by_level = {n: average_tier2_runs(runs) for n, runs in tier2_raw.items()}

    section("TIER-2 COMPLETION LATENCY  (Async — time until the deep analysis actually finishes)")
    if tier2_by_level:
        print("""
  The Tier-1 "Asynch XAI" numbers above measure only the fast acknowledgment
  (task_id returned). This section measures the real end-to-end time until
  the background SHAP-interactions + counterfactual analysis is retrievable
  via GET /result/{task_id} — i.e. what the user actually waits for.
""")
        print(f"  {'Users':>6}  {'Tier-1 p95 (ms)':>16}  {'Tier-2 p50 (ms)':>16}  {'Tier-2 p95 (ms)':>16}  {'Tier-2 p99 (ms)':>16}  {'Completions':>12}")
        for n in CONCURRENCY_LEVELS:
            tier1 = all_data["asynch"].get(n, {})
            tier2 = tier2_by_level.get(n)
            if not tier2:
                continue
            print(f"  {n:>6}  {tier1.get('p95', 0):>16,.0f}  {tier2['p50']:>16,.0f}  {tier2['p95']:>16,.0f}  {tier2['p99']:>16,.0f}  {tier2['request_count']:>12}")
        print()
    else:
        print("\n  No tier2_completion data found — locust_asynch.py may predate the polling\n  instrumentation, or no Medium/High risk requests occurred.\n")

    sys.stdout = sys.__stdout__
    report_file.close()
    print(f"Report saved  → {REPORT_PATH}")

    # ── Failure breakdown (separate file, diagnostic only) ─────────────────────
    failure_data = {arch: collect_failure_reasons(arch) for arch in ARCHS}
    write_failure_breakdown(failure_data)

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_p95_latency(all_data)
    plot_percentiles(all_data)
    plot_throughput(all_data)
    plot_failure_rate(all_data)
    plot_boxplots(p95_samples)
    plot_p95_heatmap(all_data)
    plot_speedup_ratio(all_data)
    print(f"All plots  →  {PLOTS_DIR}/")
    print("Done.")
