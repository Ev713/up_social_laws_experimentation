import argparse
import csv
from pathlib import Path

from experimentation.robustness_runner import (
    ProgressReporter,
    VERIFIER_SPECS,
    build_cases,
    compile_for_verifier,
    load_problem,
    parse_config,
)


def main():
    parser = argparse.ArgumentParser(description="Load all configured problems and verify they compile for the selected verifiers.")
    parser.add_argument("config", help="Path to experiment config JSON.")
    args = parser.parse_args()

    config = parse_config(Path(args.config).resolve())
    cases, warnings = build_cases(config)

    out_dir = config.output_dir / f"{config.run_id}__loading_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "loading_results.csv"
    warnings_path = out_dir / "warnings.txt"
    progress_log = out_dir / "progress.log"
    warnings_path.write_text("\n".join(warnings) + ("\n" if warnings else ""))
    if progress_log.exists():
        progress_log.unlink()

    total_pairs = len(cases) * len(config.verifiers)
    reporter = ProgressReporter(progress_log)
    reporter.emit(
        f"Starting loading check '{config.run_id}' with {len(cases)} cases and "
        f"{len(config.verifiers)} verifiers ({total_pairs} case/verifier pairs)."
    )
    reporter.emit(f"Output directory: {out_dir}")
    reporter.emit(f"Loading results CSV: {csv_path}")
    reporter.emit(f"Warnings file: {warnings_path}")
    if warnings:
        reporter.emit(f"Case generation warnings: {len(warnings)}")

    failures = 0
    completed = 0
    try:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "verifier",
                    "status",
                    "problem_name",
                    "compiled_problem_name",
                    "details",
                ],
            )
            writer.writeheader()
            for case_index, case in enumerate(cases, start=1):
                reporter.emit(
                    f"Case {case_index}/{len(cases)}: domain={case.domain}, instance={case.instance_file}, "
                    f"social_law={case.social_law_label}"
                )
                for verifier_index, verifier_label in enumerate(config.verifiers, start=1):
                    completed += 1
                    reporter.emit(
                        f"Starting pair {completed}/{total_pairs}: case_id={case.case_id}, "
                        f"verifier={verifier_label} ({verifier_index}/{len(config.verifiers)} for this case)"
                    )
                    verifier = VERIFIER_SPECS[verifier_label]
                    try:
                        problem = load_problem(case)
                        _, compiled_problem, _ = compile_for_verifier(problem, verifier)
                        writer.writerow({
                            "case_id": case.case_id,
                            "verifier": verifier_label,
                            "status": "OK",
                            "problem_name": problem.name,
                            "compiled_problem_name": compiled_problem.name,
                            "details": "",
                        })
                        reporter.emit(
                            f"Finished pair {completed}/{total_pairs}: case_id={case.case_id}, "
                            f"verifier={verifier_label}, status=OK, "
                            f"problem_name={problem.name}, compiled_problem_name={compiled_problem.name}"
                        )
                    except Exception as exc:
                        failures += 1
                        details = f"{type(exc).__name__}: {exc}"
                        writer.writerow({
                            "case_id": case.case_id,
                            "verifier": verifier_label,
                            "status": "FAIL",
                            "problem_name": "",
                            "compiled_problem_name": "",
                            "details": details,
                        })
                        reporter.emit(
                            f"Finished pair {completed}/{total_pairs}: case_id={case.case_id}, "
                            f"verifier={verifier_label}, status=FAIL, details={details}"
                        )
                    f.flush()
    finally:
        reporter.emit(f"Loading check stopped after {completed}/{total_pairs} pairs.")
        reporter.emit(f"Loading results: {csv_path}")
        if warnings:
            reporter.emit(f"Warnings: {warnings_path}")
        reporter.emit(f"Failures so far: {failures}")
        reporter.close()

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
