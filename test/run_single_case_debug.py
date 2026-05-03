import argparse
from pathlib import Path

from experimentation.robustness_runner import ProblemCase, ResourceLimits, evaluate_problem


def main():
    parser = argparse.ArgumentParser(description="Run one case/verifier pair and write an easy-to-read debug log.")
    parser.add_argument("domain")
    parser.add_argument("instance")
    parser.add_argument("verifier", choices=["general", "simple"])
    parser.add_argument("--with-sl", action="store_true", dest="with_sl")
    parser.add_argument("--engine", default="enhsp")
    parser.add_argument("--planner-timeout", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    case = ProblemCase(domain=args.domain, instance_file=args.instance, has_social_law=args.with_sl)
    limits = ResourceLimits(engine=args.engine, planner_timeout_seconds=args.planner_timeout)
    result = evaluate_problem(case, args.verifier, limits)

    output = Path(args.output) if args.output else Path("test/.logs") / f"{case.case_id}__{args.verifier}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.pop("log_text"))

    print(f"log: {output}")
    for key in [
        "status",
        "planner_status",
        "single_agent_statuses",
        "source_problem_name",
        "compiled_problem_name",
        "warnings",
    ]:
        print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()
