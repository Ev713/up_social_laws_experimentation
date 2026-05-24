import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimentation.robustness_runner import ProblemCase, VERIFIER_SPECS, compile_for_verifier, load_problem
from unified_planning.model import FNode
from unified_planning.model.action import InstantaneousAction
from unified_planning.plans import ActionInstance
from unified_planning.shortcuts import SequentialSimulator


def objects_for_type(problem, param_type):
    return list(problem.objects(param_type))


def ground_action_instances(problem, action):
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


def applicable_action_instances(simulator, state, grounded_actions):
    applicable = []
    for ai in grounded_actions:
        if simulator.is_applicable(state, ai.action, ai.actual_parameters):
            applicable.append(ai)
    applicable.sort(key=format_action_instance)
    return applicable


def format_action_instance(ai: ActionInstance) -> str:
    params = ", ".join(obj.name for obj in ai.actual_parameters)
    return f"{ai.action.name}({params})" if params else ai.action.name


def iter_ground_fluents(problem):
    for fluent in problem.fluents:
        domains = [objects_for_type(problem, p.type) for p in fluent.signature]
        if not domains:
            yield fluent()
            continue
        for params in itertools.product(*domains):
            yield fluent(*params)


def summarize_fnode_value(value: FNode) -> str:
    try:
        return str(value.constant_value())
    except Exception:
        return str(value)


def should_show_value(exp: FNode, value: FNode) -> bool:
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
        if patterns and not any(p in label for p in patterns):
            continue
        value = state.get_value(exp)
        if not patterns and not should_show_value(exp, value):
            continue
        print(f"  {label} = {summarize_fnode_value(value)}")
        shown += 1
    if shown == 0:
        print("  <nothing to show>")


def print_effects(ai: ActionInstance):
    print(f"Effects for {format_action_instance(ai)}:")
    for eff in ai.action.effects:
        print(f"  {eff}")


def main():
    parser = argparse.ArgumentParser(description="Interactive step-by-step simulator for a compiled robustness problem.")
    parser.add_argument("domain", help="Domain name as used by experimentation.robustness_runner")
    parser.add_argument("instance", help="Instance file name, e.g. pfile1.json")
    parser.add_argument("--with-sl", action="store_true", dest="with_sl", help="Load the instance with social law enabled")
    parser.add_argument("--verifier", choices=["general", "simple"], default="general", help="Compile the problem with this verifier before simulation")
    parser.add_argument("--show-state", action="store_true", help="Print the non-zero/true state at each step")
    parser.add_argument("--show-effects", action="store_true", help="Print the chosen action schema effects before applying it")
    parser.add_argument("--track", action="append", default=[], help="Substring of fluent names to always print from the state; can be used multiple times")
    args = parser.parse_args()

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
        for idx, ai in enumerate(applicable):
            print(f"  [{idx}] {format_action_instance(ai)}")

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

        ai = applicable[idx]
        print(f"Chosen action: {format_action_instance(ai)}")
        if args.show_effects:
            print_effects(ai)

        new_state = simulator.apply(state, ai.action, ai.actual_parameters)
        if new_state is None:
            print("Action application failed.")
            continue
        state = new_state
        step += 1


if __name__ == "__main__":
    main()
