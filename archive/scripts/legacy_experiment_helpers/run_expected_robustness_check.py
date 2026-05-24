import argparse
import csv
from pathlib import Path

from experimentation.robustness_runner import build_runner_from_config_path


NON_ROBUST_MULTI_AGENT = {
    "NON_ROBUST_MULTI_AGENT_FAIL",
    "NON_ROBUST_MULTI_AGENT_DEADLOCK",
}

TIMEOUT_PLANNER_STATUSES = {
    "TIMEOUT",
    "WALL_TIMEOUT",
}


def main():
    parser = argparse.ArgumentParser(description="Run the configured suite and check expected robustness outcomes.")
    parser.add_argument("config", help="Path to experiment config JSON.")
    args = parser.parse_args()

    runner = build_runner_from_config_path(Path(args.config).resolve())
    results = runner.run()

    checks_path = runner.run_dir / "expectation_checks.csv"
    warnings_path = runner.run_dir / "expectation_warnings.txt"
    warning_lines = []
    failures = 0

    with checks_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "verifier",
                "status",
                "planner_status",
                "expectation",
                "outcome",
                "details",
            ],
        )
        writer.writeheader()

        for result in results:
            has_social_law = bool(result["has_social_law"])
            status = str(result["status"])
            planner_status = str(result["planner_status"])

            if not has_social_law:
                expectation = "multi_agent_non_robust"
                if status in NON_ROBUST_MULTI_AGENT:
                    outcome = "PASS"
                    details = ""
                elif status == "UNKNOWN" and planner_status in TIMEOUT_PLANNER_STATUSES:
                    outcome = "WARNING"
                    details = "Timeout hit; maybe raise limits for this test run."
                    warning_lines.append(
                        f"{result['case_id']} ({result['verifier']}): timeout while expecting non-robust without social law."
                    )
                else:
                    outcome = "FAIL"
                    details = "Expected multi-agent non-robust result without social law."
                    failures += 1
            else:
                expectation = "robust"
                if status == "ROBUST_RATIONAL":
                    outcome = "PASS"
                    details = ""
                elif status == "UNKNOWN" and planner_status in TIMEOUT_PLANNER_STATUSES:
                    outcome = "WARNING"
                    details = "Timeout hit; maybe raise limits for this test run."
                    warning_lines.append(
                        f"{result['case_id']} ({result['verifier']}): timeout while expecting robust with social law."
                    )
                else:
                    outcome = "FAIL"
                    details = "Expected robust result with social law."
                    failures += 1

            writer.writerow({
                "case_id": result["case_id"],
                "verifier": result["verifier"],
                "status": status,
                "planner_status": planner_status,
                "expectation": expectation,
                "outcome": outcome,
                "details": details,
            })

    warnings_path.write_text("\n".join(warning_lines) + ("\n" if warning_lines else ""))
    print(f"Expectation checks: {checks_path}")
    if warning_lines:
        print(f"Warnings: {warnings_path}")
    if failures:
        print(f"Expectation check failed for {failures} cases.")
        raise SystemExit(1)
    print("Expectation check passed.")


if __name__ == "__main__":
    main()
