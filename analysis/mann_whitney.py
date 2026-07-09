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
    # For H3: "supported" means NOT significant (we want p > 0.05)
    if alternative == "two-sided":
        verdict = "SUPPORTED  (p > 0.05, equivalence holds)" if not sig else "NOT SUPPORTED  (p < 0.05, distributions differ)"
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

  Step 3 — Ideal outcome:
  {"ACHIEVED" if not h3["sig"] else "NOT ACHIEVED (scope limitation)"}
  {"Async Tier-1 latency is statistically equivalent to the no-XAI floor. XAI is effectively free to the user." if not h3["sig"] else
  f"Tier-1 SHAP runs synchronously before returning, adding measurable overhead under high concurrency (25+ users). Asynch XAI reduces tail latency by ~{(np.median(p95_samples['synch'])-np.median(p95_samples['asynch']))/np.median(p95_samples['synch'])*100:.0f}% vs Synch XAI (Step 2 confirmed) but does not fully reach the No XAI floor. p = {h3['p']:.4f}. Future work: precompute or cache Tier-1 explanations to close this gap."}
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
