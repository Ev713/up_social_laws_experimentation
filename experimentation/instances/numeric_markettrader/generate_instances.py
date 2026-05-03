import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def market_names(count: int):
    return [f"m{i}" for i in range(count)]


def goods_names(count: int):
    return [f"g{i}" for i in range(count)]


def agent_names(count: int):
    return [f"camel{i}" for i in range(count)]


def positive_price(level: int, good_idx: int, market_idx: int) -> int:
    return 2 + ((level + good_idx + market_idx) % 4)


def sell_price(level: int, buy_price: int, market_idx: int, source_idx: int, num_markets: int) -> int:
    # Economic structure: the source market is break-even, one market is a bad deal,
    # and one or two nearby markets are profitable resale destinations.
    if market_idx == source_idx:
        return buy_price
    loss_market = (source_idx + num_markets - 1) % num_markets
    profitable_count = 1 if num_markets <= 3 else 2
    profitable_markets = {((source_idx + offset) % num_markets) for offset in range(1, profitable_count + 1)}
    if market_idx in profitable_markets:
        return buy_price + 2
    if market_idx == loss_market:
        return max(1, buy_price - 2)
    return max(1, buy_price - 1)


def drive_cost(level: int, from_idx: int, to_idx: int) -> int:
    return 1 + ((from_idx + to_idx) % 2)


def price_entries(data):
    return [
        entry
        for entry in data["init_values"]["global"]
        if entry[0] == "=" and entry[1][0] == "price"
    ]


def sellprice_entries(data):
    return [
        entry
        for entry in data["init_values"]["global"]
        if entry[0] == "=" and entry[1][0] == "sellprice"
    ]


def stock_entries(data):
    return [
        entry
        for entry in data["init_values"]["global"]
        if entry[0] == "=" and entry[1][0] == "on-sale"
    ]


def drive_cost_entries(data):
    return [
        entry
        for entry in data["init_values"]["global"]
        if entry[0] == "=" and entry[1][0] == "drive-cost"
    ]


def make_more_conflict_inducing(data, rng):
    """
    Apply one mutation that is specifically aimed at increasing interaction
    between agents rather than just making goals easier.

    In this domain that means making the scarce shared source matter more and
    making profitable routes cheaper to traverse:
    - reduce available stock at a source market (never below 1)
    - lower one travel cost (never below 1)
    - lower a source buy price so multiple agents prefer the same source

    This still does not change the domain model or target solver runtimes.
    """
    choice = rng.choice(["stock", "travel", "source_buy"])
    if choice == "stock":
        entry = rng.choice(stock_entries(data))
        entry[2] = str(max(1, int(entry[2]) - 1))
        return f"reduced stock for {entry[1][1]}"
    if choice == "travel":
        entry = rng.choice(drive_cost_entries(data))
        entry[2] = str(max(1, int(entry[2]) - 1))
        return f"lowered travel cost for {entry[1][1]}"
    source_prices = [
        entry for entry in price_entries(data)
        if any(stock[1][1][0] == entry[1][1][0] and stock[1][1][1] == entry[1][1][1] for stock in stock_entries(data))
    ]
    entry = rng.choice(source_prices)
    entry[2] = str(max(1, int(entry[2]) - 1))
    return f"lowered source price for {entry[1][1]}"


def make_easier(data, rng):
    """
    Apply one random easing mutation to an instance in-place.

    This only changes economic parameters that already exist in the JSON:
    - increases initial stock for one source market
    - raises one sell price
    - lowers one buy price

    It does not:
    - add or remove agents, markets, or goods
    - change the action model
    - inspect planner runtimes or optimize for timeouts

    The intended use is transparent calibration toward semantic targets
    such as single-agent solvability and non-robust multi-agent behavior.
    """
    choice = rng.choice(["stock", "sell", "buy"])
    if choice == "stock":
        entry = rng.choice(stock_entries(data))
        entry[2] = str(int(entry[2]) + 1)
        return f"increased stock for {entry[1][1]}"
    if choice == "sell":
        entry = rng.choice(sellprice_entries(data))
        entry[2] = str(int(entry[2]) + 1)
        return f"raised sellprice for {entry[1][1]}"
    entry = rng.choice(price_entries(data))
    entry[2] = str(max(1, int(entry[2]) - 1))
    return f"lowered price for {entry[1][1]}"


def make_harder(data, rng):
    """
    Apply one random hardening mutation to an instance in-place.

    This is the inverse of make_easier():
    - decreases initial stock for one source market, but never below 1
    - lowers one sell price, but never below 1
    - raises one buy price

    This is meant for bounded calibration around a target difficulty band.
    It is not meant to manufacture timeout-heavy instances or to tune against
    a specific planner's search behavior.
    """
    choice = rng.choice(["stock", "sell", "buy"])
    if choice == "stock":
        entry = rng.choice(stock_entries(data))
        entry[2] = str(max(1, int(entry[2]) - 1))
        return f"decreased stock for {entry[1][1]}"
    if choice == "sell":
        entry = rng.choice(sellprice_entries(data))
        entry[2] = str(max(1, int(entry[2]) - 1))
        return f"lowered sellprice for {entry[1][1]}"
    entry = rng.choice(price_entries(data))
    entry[2] = str(int(entry[2]) + 1)
    return f"raised price for {entry[1][1]}"


def calibrate_instance(data, outcome, rng):
    """
    Return a single bounded calibration step based on a semantic outcome label.

    Supported interpretation:
    - if the instance is robust, make it easier to introduce contention
    - if it is single-agent unsolvable, make it easier to restore solvability
    - if it is already non-robust multi-agent, keep it unchanged
    - otherwise, leave it unchanged

    What this does:
    - mutates only instance parameters
    - returns a short textual mutation description for logging

    What this does not do:
    - modify the domain model
    - use runtime/timeout feedback as a target
    - claim to prove anything by itself; it is only a mutation operator
      to be used by an external calibration loop that evaluates semantics
      with the robustness checker
    """
    if outcome == "ROBUST_RATIONAL":
        return make_more_conflict_inducing(data, rng)
    if outcome == "NON_ROBUST_SINGLE_AGENT":
        return make_easier(data, rng)
    if outcome == "NON_ROBUST_MULTI_AGENT_FAIL":
        return "kept instance unchanged"
    return "kept instance unchanged"


def make_instance(level: int):
    num_agents = min(4, 2 + (level - 1) // 7)
    num_markets = 3 + (level - 1) // 2
    num_goods = 1 + (level - 1) // 3

    markets = market_names(num_markets)
    goods = goods_names(num_goods)
    agents = agent_names(num_agents)

    init_global = []
    max_initial_stock = 0

    for good_idx, good in enumerate(goods):
        source_idx = good_idx % num_markets
        source_market = markets[source_idx]
        initial_stock = max(2, num_agents + (level + good_idx) % 2)
        max_initial_stock = max(max_initial_stock, initial_stock)
        init_global.append(["=", ["on-sale", [good, source_market]], str(initial_stock)])

        for market_idx, market in enumerate(markets):
            if market_idx == source_idx:
                buy = 2 + (good_idx % 2)
            else:
                buy = positive_price(level, good_idx, market_idx)
            sell = sell_price(level, buy, market_idx, source_idx, num_markets)
            init_global.append(["=", ["price", [good, market]], str(buy)])
            init_global.append(["=", ["sellprice", [good, market]], str(sell)])

    for from_idx, from_market in enumerate(markets):
        for to_idx, to_market in enumerate(markets):
            if from_idx == to_idx:
                continue
            init_global.append(
                ["=", ["drive-cost", [from_market, to_market]], str(drive_cost(level, from_idx, to_idx))]
            )

    init_values = {"global": init_global}
    goals = {"global": []}

    base_cash = 16 + level * 2
    base_capacity = 3 + min(2, level // 6)

    for agent_idx, agent in enumerate(agents):
        start_market = markets[0]
        values = [
            ["at", [start_market]],
            ["=", ["cash", []], str(base_cash)],
            ["=", ["capacity", []], str(base_capacity)],
        ]
        for good in goods:
            values.append(["=", ["bought", [good]], "0"])
        for from_market in markets:
            for to_market in markets:
                if from_market == to_market:
                    continue
                values.append(["can-drive", [from_market, to_market]])
        init_values[agent] = values

        target_cash = base_cash + 2 + min(level // 6, 2)
        goals[agent] = [[">=", ["cash", []], str(target_cash)]]

    return {
        "market": markets,
        "goods": goods,
        "agents": agents,
        "init_values": init_values,
        "bounds": {
            "inventory": max_initial_stock,
            "capacity": base_capacity + 20,
        },
        "goals": goals,
    }



def write_instance(path: Path, data):
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def generate_all_instances():
    for level in range(1, 21):
        write_instance(ROOT / f"pfile{level}.json", make_instance(level))


def _ensure_project_root_on_path():
    project_root = ROOT.parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def evaluate_instance_data(instance_name: str, data, planner_timeout_seconds: int = 60):
    _ensure_project_root_on_path()
    from experimentation.robustness_runner import ProblemCase, ResourceLimits, evaluate_problem

    tmp_name = f"__calibration__{instance_name}"
    tmp_path = ROOT / tmp_name
    write_instance(tmp_path, data)
    try:
        return evaluate_problem(
            ProblemCase(domain="numeric_markettrader", instance_file=tmp_name, has_social_law=False),
            verifier_label="general",
            limits=ResourceLimits(
                engine="enhsp",
                planner_timeout_seconds=planner_timeout_seconds,
                wall_timeout_seconds=max(planner_timeout_seconds, 60),
                cpu_seconds=max(planner_timeout_seconds, 60),
                memory_bytes=16_000_000_000,
            ),
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# Calibration is intentionally bounded and transparent.
# It uses semantic outcome labels from the robustness checker, not timeout behavior.
# It does not alter the domain model, only numeric parameters in the JSON instances.
# It records every mutation and the observed outcome after that mutation in a sidecar log.
def calibrate_instances(instance_numbers=None, seed: int = 0, max_steps: int = 20, planner_timeout_seconds: int = 60):
    if instance_numbers is None:
        instance_numbers = list(range(1, 21))

    rng = random.Random(seed)
    summaries = []
    for idx in instance_numbers:
        path = ROOT / f"pfile{idx}.json"
        data = json.loads(path.read_text())
        evaluation = evaluate_instance_data(path.name, data, planner_timeout_seconds=planner_timeout_seconds)
        outcome = evaluation["status"]
        history = {
            "instance": path.name,
            "seed": seed,
            "max_steps": max_steps,
            "planner_timeout_seconds": planner_timeout_seconds,
            "initial_status": outcome,
            "steps": [],
        }

        for step in range(max_steps):
            if outcome == "NON_ROBUST_MULTI_AGENT_FAIL":
                break
            mutation = calibrate_instance(data, outcome, rng)
            evaluation = evaluate_instance_data(path.name, data, planner_timeout_seconds=planner_timeout_seconds)
            outcome = evaluation["status"]
            history["steps"].append({
                "step": step + 1,
                "mutation": mutation,
                "status": outcome,
                "planner_status": evaluation["planner_status"],
            })
            if outcome == "NON_ROBUST_MULTI_AGENT_FAIL":
                break

        history["final_status"] = outcome
        history["final_planner_status"] = evaluation["planner_status"]
        write_instance(path, data)
        (ROOT / f"pfile{idx}.calibration.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        summaries.append((path.name, history["initial_status"], history["final_status"], len(history["steps"])))

    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true", help="Calibrate existing instances toward the non-robust multi-agent regime.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for bounded calibration mutations.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum calibration mutations per instance.")
    parser.add_argument("--timeout", type=int, default=60, help="Planner timeout in seconds for each semantic calibration check.")
    parser.add_argument("instances", nargs="*", type=int, help="Optional instance numbers to generate or calibrate.")
    args = parser.parse_args()

    if args.calibrate:
        instance_numbers = args.instances or list(range(1, 21))
        summaries = calibrate_instances(
            instance_numbers=instance_numbers,
            seed=args.seed,
            max_steps=args.max_steps,
            planner_timeout_seconds=args.timeout,
        )
        for name, initial_status, final_status, steps in summaries:
            print(f"{name}: {initial_status} -> {final_status} in {steps} steps")
        return

    if args.instances:
        for level in args.instances:
            write_instance(ROOT / f"pfile{level}.json", make_instance(level))
    else:
        generate_all_instances()


if __name__ == "__main__":
    main()
