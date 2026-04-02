import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def build_exclusion_suffix(excluded_domains: Iterable[str]) -> str:
    cleaned = [domain for domain in excluded_domains if domain]
    if not cleaned:
        return ""
    return "__excluding_" + "_and_".join(cleaned)


TERMINAL_PLANNER_STATUSES = {
    "SOLVED_SATISFICING",
    "SOLVED_OPTIMALLY",
    "UNSOLVABLE_PROVEN",
    "UNSOLVABLE_INCOMPLETELY",
}
UNSOLVED_PLANNER_STATUSES = {
    "TIMEOUT",
    "WALL_TIMEOUT",
    "INTERNAL_ERROR",
    "WORKER_ERROR",
    "NO_RESULT",
}


@dataclass
class LogMetrics:
    path: str
    oom_detected: bool = False
    weird_wt_phi_warnings: int = 0
    final_expanded_nodes: Optional[int] = None
    final_states_evaluated: Optional[int] = None
    max_expanded_nodes: Optional[int] = None
    max_states_evaluated: Optional[int] = None
    max_avg_speed: Optional[float] = None
    final_reported_time_seconds: Optional[float] = None
    grounding_time_ms: Optional[int] = None
    planning_time_ms: Optional[int] = None
    heuristic_time_ms: Optional[int] = None
    search_time_ms: Optional[int] = None
    num_fluents: Optional[int] = None
    num_numeric_variables: Optional[int] = None
    num_actions: Optional[int] = None
    num_processes: Optional[int] = None
    num_events: Optional[int] = None


def _read_csv_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if None in row:
                continue
            rows.append(row)
    return rows


def load_results(results_csv: Path) -> pd.DataFrame:
    rows = _read_csv_rows(results_csv)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["has_social_law"] = df["has_social_law"].map({"True": True, "False": False})
    df["social_law_label"] = df["has_social_law"].map({True: "with_sl", False: "without_sl"})
    df["elapsed_seconds"] = pd.to_numeric(df["elapsed_seconds"], errors="coerce")
    df["completed_before_timeout"] = ~df["planner_status"].isin(UNSOLVED_PLANNER_STATUSES)
    df["timed_out"] = df["planner_status"].isin({"TIMEOUT", "WALL_TIMEOUT"})
    df["internal_error"] = df["planner_status"].eq("INTERNAL_ERROR")
    df["non_robust_single_agent"] = df["status"].eq("NON_ROBUST_SINGLE_AGENT")
    df["instance_number"] = df["instance"].str.extract(r"pfile(\d+)").astype(float)
    df["log_file_name"] = df["log_file"].map(lambda value: Path(value).name)
    return df


def parse_progress_log(progress_path: Path) -> pd.DataFrame:
    if not progress_path.exists():
        return pd.DataFrame()

    pattern = re.compile(
        r"Finished pair (?P<pair_done>\d+)/(?P<pair_total>\d+): case_id=(?P<case_id>[^,]+), "
        r"verifier=(?P<verifier>[^,]+), status=(?P<status>[^,]+), "
        r"planner_status=(?P<planner_status>[^,]+), elapsed_seconds=(?P<elapsed_seconds>[^,]+), "
        r"wall_clock_seconds=(?P<wall_clock_seconds>[^,]+), log_file=(?P<log_file>.+)$"
    )
    rows: List[dict] = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        row = match.groupdict()
        row["pair_done"] = int(row["pair_done"])
        row["pair_total"] = int(row["pair_total"])
        row["elapsed_seconds"] = float(row["elapsed_seconds"])
        row["wall_clock_seconds"] = float(row["wall_clock_seconds"])
        rows.append(row)
    return pd.DataFrame(rows)


def parse_log_metrics(log_path: Path) -> LogMetrics:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = LogMetrics(path=str(log_path))
    metrics.oom_detected = "OutOfMemoryError" in text
    metrics.weird_wt_phi_warnings = text.count("Warning, weird things with wt^phi happen")

    def extract_last_int(pattern: str) -> Optional[int]:
        matches = re.findall(pattern, text)
        return int(matches[-1]) if matches else None

    def extract_last_float(pattern: str) -> Optional[float]:
        matches = re.findall(pattern, text)
        return float(matches[-1]) if matches else None

    metrics.final_expanded_nodes = extract_last_int(r"Expanded Nodes:(\d+)")
    metrics.final_states_evaluated = extract_last_int(r"States Evaluated:(\d+)")
    metrics.grounding_time_ms = extract_last_int(r"Grounding Time:\s*(\d+)")
    metrics.planning_time_ms = extract_last_int(r"Planning Time \(msec\):\s*(\d+)")
    metrics.heuristic_time_ms = extract_last_int(r"Heuristic Time \(msec\):\s*(\d+)")
    metrics.search_time_ms = extract_last_int(r"Search Time \(msec\):\s*(\d+)")
    metrics.num_fluents = extract_last_int(r"\|F\|:(\d+)")
    metrics.num_numeric_variables = extract_last_int(r"\|X\|:(\d+)")
    metrics.num_actions = extract_last_int(r"\|A\|:(\d+)")
    metrics.num_processes = extract_last_int(r"\|P\|:(\d+)")
    metrics.num_events = extract_last_int(r"\|E\|:(\d+)")

    periodic_matches = re.findall(
        r"Time:\s*([0-9.]+)s\s*;\s*Expanded Nodes:\s*([0-9]+)\s*\(Avg-Speed\s*([0-9.]+)\s*n/s\)\s*;\s*Evaluated States:\s*([0-9]+)",
        text,
    )
    if periodic_matches:
        times = [float(match[0]) for match in periodic_matches]
        expanded = [int(match[1]) for match in periodic_matches]
        speeds = [float(match[2]) for match in periodic_matches]
        evaluated = [int(match[3]) for match in periodic_matches]
        metrics.final_reported_time_seconds = times[-1]
        metrics.max_expanded_nodes = max(expanded)
        metrics.max_states_evaluated = max(evaluated)
        metrics.max_avg_speed = max(speeds)
        if metrics.final_expanded_nodes is None:
            metrics.final_expanded_nodes = expanded[-1]
        if metrics.final_states_evaluated is None:
            metrics.final_states_evaluated = evaluated[-1]

    return metrics


def load_log_metrics(log_dir: Path) -> pd.DataFrame:
    rows = []
    for log_path in sorted(log_dir.glob("*.log")):
        row = parse_log_metrics(log_path).__dict__
        row["log_file_name"] = log_path.name
        rows.append(row)
    return pd.DataFrame(rows)


def cumulative_counts(times: Iterable[float], thresholds: Iterable[float]) -> List[int]:
    sorted_times = sorted(t for t in times if pd.notna(t))
    counts = []
    idx = 0
    for threshold in thresholds:
        while idx < len(sorted_times) and sorted_times[idx] <= threshold:
            idx += 1
        counts.append(idx)
    return counts


def save_plot_data(df: pd.DataFrame, output_path: Path):
    df.to_csv(output_path, index=False)


def plot_solved_under_time(results_df: pd.DataFrame, output_dir: Path, file_suffix: str = ""):
    solved_df = results_df[results_df["completed_before_timeout"]].copy()
    if solved_df.empty:
        return

    max_elapsed = solved_df["elapsed_seconds"].max()
    upper = max(60, int(math.ceil(max_elapsed / 60.0) * 60))
    thresholds = list(range(0, upper + 1, 30))

    for verifier in sorted(solved_df["verifier"].unique()):
        for social_law_label in ["without_sl", "with_sl"]:
            subset = solved_df[
                (solved_df["verifier"] == verifier)
                & (solved_df["social_law_label"] == social_law_label)
            ]
            if subset.empty:
                continue

            plot_rows = []
            plt.figure(figsize=(11, 7)) if plt else None
            for domain in sorted(subset["domain"].unique()):
                domain_times = subset.loc[subset["domain"] == domain, "elapsed_seconds"].tolist()
                counts = cumulative_counts(domain_times, thresholds)
                for threshold, count in zip(thresholds, counts):
                    plot_rows.append({
                        "verifier": verifier,
                        "social_law_label": social_law_label,
                        "domain": domain,
                        "threshold_seconds": threshold,
                        "solved_count": count,
                    })
                if plt:
                    plt.plot(thresholds, counts, label=domain, linewidth=2)

            plot_df = pd.DataFrame(plot_rows)
            base_name = f"solved_under_time__{verifier}__{social_law_label}{file_suffix}"
            save_plot_data(plot_df, output_dir / f"{base_name}.csv")
            if plt:
                plt.title(f"Solved within x seconds: verifier={verifier}, social_law={social_law_label}")
                plt.xlabel("x (seconds)")
                plt.ylabel("Solved count")
                plt.grid(True, alpha=0.3)
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                plt.savefig(output_dir / f"{base_name}.png", dpi=200)
                plt.close()


def plot_log_metric_scatter(merged_df: pd.DataFrame, output_dir: Path):
    metric_specs = [
        ("final_expanded_nodes", "Expanded Nodes"),
        ("final_states_evaluated", "States Evaluated"),
    ]
    if merged_df.empty:
        return

    for metric, label in metric_specs:
        subset = merged_df[merged_df[metric].notna() & merged_df["elapsed_seconds"].notna()].copy()
        if subset.empty:
            continue

        csv_name = f"{metric}_vs_elapsed.csv"
        subset[
            [
                "case_id",
                "domain",
                "verifier",
                "social_law_label",
                "planner_status",
                "status",
                "elapsed_seconds",
                metric,
                "oom_detected",
                "weird_wt_phi_warnings",
            ]
        ].to_csv(output_dir / csv_name, index=False)

        if plt:
            plt.figure(figsize=(11, 7))
            statuses = sorted(subset["planner_status"].dropna().unique())
            for planner_status in statuses:
                status_subset = subset[subset["planner_status"] == planner_status]
                plt.scatter(
                    status_subset["elapsed_seconds"],
                    status_subset[metric],
                    label=planner_status,
                    alpha=0.8,
                    s=45,
                )
            plt.xlabel("Elapsed seconds")
            plt.ylabel(label)
            plt.title(f"{label} vs elapsed time")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / f"{metric}_vs_elapsed.png", dpi=200)
            plt.close()


def build_latex_table(results_df: pd.DataFrame, output_path: Path):
    completed_df = results_df[results_df["completed_before_timeout"]].copy()
    grouped = (
        completed_df.groupby(["domain", "social_law_label", "verifier"])
        .size()
        .unstack(["social_law_label", "verifier"], fill_value=0)
        .sort_index()
    )

    ordered_columns = []
    for social_law_label in ["without_sl", "with_sl"]:
        for verifier in sorted(results_df["verifier"].unique()):
            column = (social_law_label, verifier)
            if column in grouped.columns:
                ordered_columns.append(column)
    grouped = grouped.reindex(columns=ordered_columns, fill_value=0)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\\begin{tabular}{l" + "r" * len(grouped.columns) + "}\n")
        handle.write("\\toprule\n")
        header_labels = ["Domain"] + [f"{sl.replace('_', '\\_')} {verifier}" for sl, verifier in grouped.columns]
        handle.write(" & ".join(header_labels) + " \\\\\n")
        handle.write("\\midrule\n")
        for domain, row in grouped.iterrows():
            values = [str(int(row[col])) for col in grouped.columns]
            handle.write(domain.replace("_", "\\_") + " & " + " & ".join(values) + " \\\\\n")
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def build_bug_reports(merged_df: pd.DataFrame, output_dir: Path):
    internal_errors = merged_df[merged_df["internal_error"]].copy()
    internal_errors.to_csv(output_dir / "internal_errors.csv", index=False)

    single_agent_bugs = merged_df[merged_df["non_robust_single_agent"]].copy()
    single_agent_bugs.to_csv(output_dir / "non_robust_single_agent.csv", index=False)

    suspicious = merged_df[
        merged_df["internal_error"]
        | merged_df["non_robust_single_agent"]
        | merged_df["oom_detected"].eq(True)
        | (merged_df["weird_wt_phi_warnings"].fillna(0) > 0)
    ].copy()
    suspicious.to_csv(output_dir / "suspicious_cases.csv", index=False)


def build_summary_markdown(
    results_df: pd.DataFrame,
    progress_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: Path,
):
    lines = ["# Run Analysis", ""]
    lines.append(f"- Rows in results.csv: {len(results_df)}")
    lines.append(f"- Finished pairs in progress.log: {len(progress_df)}")
    lines.append(f"- Internal errors: {int(results_df['internal_error'].sum())}")
    lines.append(f"- NON_ROBUST_SINGLE_AGENT cases: {int(results_df['non_robust_single_agent'].sum())}")
    lines.append(f"- Timeouts: {int(results_df['timed_out'].sum())}")
    lines.append(
        f"- Log files with OutOfMemoryError: {int(merged_df['oom_detected'].eq(True).sum())}"
    )
    lines.append(
        f"- Log files with wt^phi warnings: {int((merged_df['weird_wt_phi_warnings'].fillna(0) > 0).sum())}"
    )
    lines.append("")

    if not merged_df.empty:
        lines.append("## Internal Error Notes")
        internal_df = merged_df[merged_df["internal_error"]].copy()
        if internal_df.empty:
            lines.append("- None found.")
        else:
            for _, row in internal_df.iterrows():
                notes = []
                if pd.notna(row.get("max_expanded_nodes")):
                    notes.append(f"max_expanded_nodes={int(row['max_expanded_nodes'])}")
                if pd.notna(row.get("max_states_evaluated")):
                    notes.append(f"max_states_evaluated={int(row['max_states_evaluated'])}")
                if bool(row.get("oom_detected")):
                    notes.append("OutOfMemoryError seen in log")
                if pd.notna(row.get("weird_wt_phi_warnings")) and int(row["weird_wt_phi_warnings"]) > 0:
                    notes.append(f"wt^phi warnings={int(row['weird_wt_phi_warnings'])}")
                joined_notes = ", ".join(notes) if notes else "no extra log clues"
                lines.append(
                    f"- {row['case_id']} / {row['verifier']}: planner_status={row['planner_status']}; {joined_notes}"
                )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(run_dir: Path, excluded_domains: Optional[List[str]] = None):
    results_csv = run_dir / "results.csv"
    progress_log = run_dir / "progress.log"
    logs_dir = run_dir / "logs"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not results_csv.exists():
        raise FileNotFoundError(f"Missing results file: {results_csv}")

    results_df = load_results(results_csv)
    excluded_domains = excluded_domains or []
    if excluded_domains:
        results_df = results_df[~results_df["domain"].isin(excluded_domains)].copy()
    progress_df = parse_progress_log(progress_log) if progress_log.exists() else pd.DataFrame()
    log_metrics_df = load_log_metrics(logs_dir) if logs_dir.exists() else pd.DataFrame()

    if not log_metrics_df.empty:
        merged_df = results_df.merge(log_metrics_df, how="left", on="log_file_name")
    else:
        merged_df = results_df.copy()

    results_df.to_csv(analysis_dir / "results_snapshot.csv", index=False)
    if not progress_df.empty:
        progress_df.to_csv(analysis_dir / "progress_snapshot.csv", index=False)
    if not log_metrics_df.empty:
        log_metrics_df.to_csv(analysis_dir / "log_metrics.csv", index=False)
    merged_df.to_csv(analysis_dir / "results_with_log_metrics.csv", index=False)

    suffix = build_exclusion_suffix(excluded_domains)

    table_name = f"paper_table_completed_before_timeout{suffix}.tex"
    summary_name = f"summary{suffix}.md"
    plot_note_name = f"plotting_note{suffix}.txt"

    build_latex_table(results_df, analysis_dir / table_name)
    build_bug_reports(merged_df, analysis_dir)
    build_summary_markdown(results_df, progress_df, merged_df, analysis_dir / summary_name)
    plot_solved_under_time(results_df, analysis_dir, suffix)
    plot_log_metric_scatter(merged_df, analysis_dir)

    if plt is None:
        note = (
            "matplotlib is not installed in the active Python environment. "
            "Plot source CSV files were written, but PNG plots were skipped.\n"
        )
        (analysis_dir / plot_note_name).write_text(note, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze robustness experiment outputs for plots, tables, and bug triage.")
    parser.add_argument("run_dir", help="Path to a run directory containing results.csv, progress.log, and logs/.")
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=[],
        help="Domain to exclude from the generated tables/plots. Can be passed multiple times.",
    )
    args = parser.parse_args()
    analyze_run(Path(args.run_dir).resolve(), excluded_domains=args.exclude_domain)


if __name__ == "__main__":
    main()
