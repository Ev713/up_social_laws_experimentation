import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimentation.analyze_run import cumulative_counts, load_results
from experimentation.analyze_run import build_exclusion_suffix


def build_overall_curve(results_df: pd.DataFrame) -> pd.DataFrame:
    solved_df = results_df[results_df["completed_before_timeout"]].copy()
    if solved_df.empty:
        return pd.DataFrame(
            columns=["verifier", "threshold_seconds", "solved_count"]
        )

    max_elapsed = solved_df["elapsed_seconds"].max()
    upper = max(60, int(math.ceil(max_elapsed / 60.0) * 60))
    thresholds = list(range(0, upper + 1, 30))

    rows = []
    for verifier in sorted(solved_df["verifier"].unique()):
        verifier_times = solved_df.loc[solved_df["verifier"] == verifier, "elapsed_seconds"].tolist()
        counts = cumulative_counts(verifier_times, thresholds)
        for threshold, count in zip(thresholds, counts):
            rows.append({
                "verifier": verifier,
                "threshold_seconds": threshold,
                "solved_count": count,
            })
    return pd.DataFrame(rows)


def plot_overall_curve(curve_df: pd.DataFrame, output_png: Path):
    plt.figure(figsize=(11, 7))
    for verifier in sorted(curve_df["verifier"].unique()):
        subset = curve_df[curve_df["verifier"] == verifier]
        plt.plot(
            subset["threshold_seconds"],
            subset["solved_count"],
            linewidth=2.5,
            label=verifier,
        )
    plt.title("Solved within x seconds across all domains")
    plt.xlabel("x (seconds)")
    plt.ylabel("Solved count")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_overall_curve_log_x(curve_df: pd.DataFrame, output_png: Path):
    plt.figure(figsize=(11, 7))
    positive_df = curve_df[curve_df["threshold_seconds"] > 0].copy()
    for verifier in sorted(positive_df["verifier"].unique()):
        subset = positive_df[positive_df["verifier"] == verifier]
        plt.plot(
            subset["threshold_seconds"],
            subset["solved_count"],
            linewidth=2.5,
            label=verifier,
        )
    ax = plt.gca()
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 3, 5), numticks=12))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    plt.title("Solved within x seconds across all domains (log-scale x-axis)")
    plt.xlabel("x (seconds, log scale)")
    plt.ylabel("Solved count")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot solved-under-time curves across all domains with one line per verifier."
    )
    parser.add_argument(
        "run_dir",
        help="Path to a run directory containing results.csv.",
    )
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=[],
        help="Domain to exclude from the generated overall curves. Can be passed multiple times.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results_df = load_results(run_dir / "results.csv")
    if args.exclude_domain:
        results_df = results_df[~results_df["domain"].isin(args.exclude_domain)].copy()
    curve_df = build_overall_curve(results_df)
    suffix = build_exclusion_suffix(args.exclude_domain)
    curve_csv = analysis_dir / f"solved_under_time__overall_by_verifier{suffix}.csv"
    curve_png = analysis_dir / f"solved_under_time__overall_by_verifier{suffix}.png"
    curve_log_png = analysis_dir / f"solved_under_time__overall_by_verifier{suffix}__log_x.png"
    curve_df.to_csv(curve_csv, index=False)
    if not curve_df.empty:
        plot_overall_curve(curve_df, curve_png)
        plot_overall_curve_log_x(curve_df, curve_log_png)


if __name__ == "__main__":
    main()
