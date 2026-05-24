import argparse
import csv
import itertools
import sys
from pathlib import Path

from experimentation.robustness_runner import (
    ProgressReporter,
    ProblemCase,
    ResourceLimits,
    VERIFIER_SPECS,
    build_cases,
    build_runner_from_config_path,
    compile_for_verifier,
    evaluate_problem,
    load_problem,
    parse_config,
)


NON_ROBUST_MULTI_AGENT = {
    "NON_ROBUST_MULTI_AGENT_FAIL",
    "NON_ROBUST_MULTI_AGENT_DEADLOCK",
}

TIMEOUT_PLANNER_STATUSES = {
    "TIMEOUT",
    "WALL_TIMEOUT",
}


def run_suite(args):
    runner = build_runner_from_config_path(Path(args.config).resolve())
    results = runner.run(resume=args.resume)
    print(f"Run directory: {runner.run_dir}")
    print(f"Cases: {len(results)}")
    status_counts = {}
    for result in results:
        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")


def run_expected_check(args):
    runner = build_runner_from_config_path(Path(args.config).resolve())
    results = runner.run(resume=args.resume)

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


def run_loading_check(args):
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


def run_single_case(args):
    case = ProblemCase(domain=args.domain, instance_file=args.instance, has_social_law=args.with_sl)
    limits = ResourceLimits(engine=args.engine, planner_timeout_seconds=args.planner_timeout)
    result = evaluate_problem(case, args.verifier, limits)

    output = Path(args.output) if args.output else Path("experimentation/tests/.logs") / f"{case.case_id}__{args.verifier}.log"
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


def objects_for_type(problem, param_type):
    return list(problem.objects(param_type))


def ground_action_instances(problem, action):
    from unified_planning.model.action import InstantaneousAction
    from unified_planning.plans import ActionInstance

    if not isinstance(action, InstantaneousAction):
        return []
    domains = [objects_for_type(problem, p.type) for p in action.parameters]
    if not domains:
        return [ActionInstance(action, tuple())]
    return [ActionInstance(action, tuple(params)) for params in itertools.product(*domains)]


def all_ground_action_instances(problem):
    instances = []
    for action in problem.actions:
        instances.extend(ground_action_instances(problem, action))
    return instances


def format_action_instance(action_instance):
    params = ", ".join(obj.name for obj in action_instance.actual_parameters)
    return f"{action_instance.action.name}({params})" if params else action_instance.action.name


def applicable_action_instances(simulator, state, grounded_actions):
    applicable = []
    for action_instance in grounded_actions:
        if simulator.is_applicable(state, action_instance.action, action_instance.actual_parameters):
            applicable.append(action_instance)
    applicable.sort(key=format_action_instance)
    return applicable


def iter_ground_fluents(problem):
    for fluent in problem.fluents:
        domains = [objects_for_type(problem, p.type) for p in fluent.signature]
        if not domains:
            yield fluent()
            continue
        for params in itertools.product(*domains):
            yield fluent(*params)


def summarize_fnode_value(value):
    try:
        return str(value.constant_value())
    except Exception:
        return str(value)


def should_show_value(exp, value):
    if exp.type.is_bool_type():
        try:
            return bool(value.bool_constant_value())
        except Exception:
            return str(value).lower() == "true"
    try:
        return value.constant_value() != 0
    except Exception:
        return True


def print_state(problem, state, patterns=None):
    patterns = patterns or []
    shown = 0
    for exp in iter_ground_fluents(problem):
        label = str(exp)
        if patterns and not any(pattern in label for pattern in patterns):
            continue
        value = state.get_value(exp)
        if not patterns and not should_show_value(exp, value):
            continue
        print(f"  {label} = {summarize_fnode_value(value)}")
        shown += 1
    if shown == 0:
        print("  <nothing to show>")


def print_effects(action_instance):
    print(f"Effects for {format_action_instance(action_instance)}:")
    for effect in action_instance.action.effects:
        print(f"  {effect}")


def run_simulator(args):
    from unified_planning.shortcuts import SequentialSimulator

    case = ProblemCase(domain=args.domain, instance_file=args.instance, has_social_law=args.with_sl)
    verifier = VERIFIER_SPECS[args.verifier]
    ma_problem = load_problem(case)
    _, compiled_problem, conversion_log = compile_for_verifier(ma_problem, verifier)

    simulator = SequentialSimulator(compiled_problem)
    state = simulator.get_initial_state()
    grounded_actions = all_ground_action_instances(compiled_problem)

    print(f"Loaded MA problem: {ma_problem.name}")
    print(f"Compiled problem: {compiled_problem.name}")
    print(f"Verifier: {args.verifier}")
    if conversion_log.strip():
        print("Compilation notes:")
        print(conversion_log.strip())
    print(f"Ground actions enumerated: {len(grounded_actions)}")
    print("Commands: index to apply action, s=state, t=tracked, r=refresh actions, q=quit")

    step = 0
    while True:
        if simulator.is_goal(state):
            print("Goal reached.")
            return

        print()
        print(f"Step {step}")
        if args.show_state:
            print("State:")
            print_state(compiled_problem, state)
        if args.track:
            print("Tracked:")
            print_state(compiled_problem, state, patterns=args.track)

        applicable = applicable_action_instances(simulator, state, grounded_actions)
        if not applicable:
            print("No applicable actions. Stopping.")
            return

        print("Applicable actions:")
        for idx, action_instance in enumerate(applicable):
            print(f"  [{idx}] {format_action_instance(action_instance)}")

        choice = input("Choose action index (or command): ").strip()
        if choice.lower() == "q":
            print("Exiting.")
            return
        if choice.lower() == "s":
            print("State:")
            print_state(compiled_problem, state)
            continue
        if choice.lower() == "t":
            print("Tracked:")
            print_state(compiled_problem, state, patterns=args.track)
            continue
        if choice.lower() == "r":
            continue
        if not choice.isdigit():
            print("Invalid input.")
            continue

        idx = int(choice)
        if idx < 0 or idx >= len(applicable):
            print("Index out of range.")
            continue

        action_instance = applicable[idx]
        print(f"Chosen action: {format_action_instance(action_instance)}")
        if args.show_effects:
            print_effects(action_instance)

        new_state = simulator.apply(state, action_instance.action, action_instance.actual_parameters)
        if new_state is None:
            print("Action application failed.")
            continue
        state = new_state
        step += 1


def run_analysis(args):
    from experimentation.analyze_run import analyze_run

    analyze_run(Path(args.run_dir).resolve(), excluded_domains=args.exclude_domain)


def build_parser():
    parser = argparse.ArgumentParser(description="Experimentation command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run robustness experiments from a JSON config.")
    run_parser.add_argument("config", help="Path to JSON config file.")
    run_parser.add_argument("--resume", action="store_true", help="Resume an existing run directory.")
    run_parser.set_defaults(func=run_suite)

    expected_parser = subparsers.add_parser("expected-check", help="Run a suite and validate expected outcomes.")
    expected_parser.add_argument("config", help="Path to JSON config file.")
    expected_parser.add_argument("--resume", action="store_true", help="Resume an existing run directory.")
    expected_parser.set_defaults(func=run_expected_check)

    loading_parser = subparsers.add_parser("loading-check", help="Load and compile configured cases.")
    loading_parser.add_argument("config", help="Path to JSON config file.")
    loading_parser.set_defaults(func=run_loading_check)

    single_parser = subparsers.add_parser("single-case", help="Run one case/verifier pair and write a debug log.")
    single_parser.add_argument("domain")
    single_parser.add_argument("instance")
    single_parser.add_argument("verifier", choices=["general", "simple"])
    single_parser.add_argument("--with-sl", action="store_true", dest="with_sl")
    single_parser.add_argument("--engine", default="enhsp")
    single_parser.add_argument("--planner-timeout", type=int, default=None)
    single_parser.add_argument("--output", default=None)
    single_parser.set_defaults(func=run_single_case)

    simulator_parser = subparsers.add_parser("simulate", help="Interactively simulate a compiled robustness problem.")
    simulator_parser.add_argument("domain")
    simulator_parser.add_argument("instance")
    simulator_parser.add_argument("--with-sl", action="store_true", dest="with_sl")
    simulator_parser.add_argument("--verifier", choices=["general", "simple"], default="general")
    simulator_parser.add_argument("--show-state", action="store_true")
    simulator_parser.add_argument("--show-effects", action="store_true")
    simulator_parser.add_argument("--track", action="append", default=[])
    simulator_parser.set_defaults(func=run_simulator)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment outputs.")
    analyze_parser.add_argument("run_dir", help="Path to a run directory containing results.csv, progress.log, and logs/.")
    analyze_parser.add_argument(
        "--exclude-domain",
        action="append",
        default=[],
        help="Domain to exclude from generated tables/plots. Can be passed multiple times.",
    )
    analyze_parser.set_defaults(func=run_analysis)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
