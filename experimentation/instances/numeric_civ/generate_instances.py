import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AGENTS = ["a0", "a1"]
ALL_RESOURCES = ["timber", "stone", "ore", "coal", "wood", "iron"]
RESOURCE_PREDICATES = {
    "timber": "is_timber",
    "stone": "is_stone",
    "ore": "is_ore",
    "coal": "is_coal",
    "wood": "is_wood",
    "iron": "is_iron",
}


def bounds_for(level: int, with_vehicles: bool):
    bounds = {
        "pollution": 4 + level * 2,
        "resource-use": 4 + level * 2,
        "housing": 1 + level // 4,
        "available": 4 + level,
        "labour": 6 + level * 4,
        "carts-at": 2 + level // 5,
    }
    if with_vehicles:
        bounds.update({
            "boat-capacity": 3,
            "train-capacity": 3,
            "boat-load": 4,
            "train-load": 4,
        })
    return bounds


def symmetric(edges):
    out = []
    for p1, p2 in edges:
        out.append((p1, p2))
        out.append((p2, p1))
    return out


def make_instance(
    idx,
    places,
    terrains,
    goals,
    resources=None,
    land_edges=None,
    sea_edges=None,
    trains=None,
    boats=None,
    agent_init_extra=None,
    global_init_extra=None,
):
    resources = resources or ALL_RESOURCES
    land_edges = symmetric(land_edges or [])
    sea_edges = symmetric(sea_edges or [])
    trains = trains or []
    boats = boats or []
    agent_init_extra = agent_init_extra or {}
    global_init_extra = global_init_extra or []

    init_global = []
    for place, flags in terrains.items():
        for flag in sorted(flags):
            init_global.append([flag, [place]])
    for resource in resources:
        init_global.append([RESOURCE_PREDICATES[resource], [resource]])
    for p1, p2 in land_edges:
        init_global.append(["connected-by-land", [p1, p2]])
    for p1, p2 in sea_edges:
        init_global.append(["connected-by-sea", [p1, p2]])
    init_global.append(["=", ["pollution", []], "0"])
    init_global.append(["=", ["resource-use", []], "0"])
    for place in places:
        init_global.append(["=", ["housing", [place]], "0"])
    for train in trains:
        for place in places:
            init_global.append(["=", ["train-capacity", [train, place]], "1"])
        init_global.append(["=", ["train-space-in", [train]], "3"])
    for boat in boats:
        init_global.append(["=", ["boat-space-in", [boat]], "3"])
    init_global.extend(global_init_extra)

    init_values = {"global": init_global}
    for agent in AGENTS:
        agent_values = [["=", ["labour", []], "0"]]
        agent_values.extend(agent_init_extra.get(agent, []))
        init_values[agent] = agent_values

    data = {
        "places": places,
        "boats": boats,
        "trains": trains,
        "resources": resources,
        "agents": AGENTS,
        "bounds": bounds_for(idx, bool(trains or boats)),
        "init_values": init_values,
        "goals": goals,
    }
    return data


INSTANCES = {
    2: make_instance(
        2,
        places=["p0"],
        terrains={"p0": {"woodland"}},
        resources=["timber"],
        goals={"a0": [["has-coal-stack", ["p0"]]], "a1": [["has-coal-stack", ["p0"]]]},
    ),
    3: make_instance(
        3,
        places=["p0"],
        terrains={"p0": {"woodland"}},
        resources=["timber", "wood"],
        goals={"a0": [["has-sawmill", ["p0"]]], "a1": [["has-sawmill", ["p0"]]]},
    ),
    4: make_instance(
        4,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain"}},
        resources=["timber", "stone", "wood"],
        goals={"global": [["has-quarry", ["p0"]]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-cabin", ["p0"]]]},
    ),
    5: make_instance(
        5,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain"}},
        resources=["timber", "stone", "wood"],
        goals={"global": [[">=", ["housing", ["p0"]], "1"]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-cabin", ["p0"]]]},
    ),
    6: make_instance(
        6,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}},
        goals={"global": [["has-mine", ["p0"]]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-coal-stack", ["p0"]]]},
    ),
    7: make_instance(
        7,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}},
        goals={"global": [["has-mine", ["p0"]], ["has-ironworks", ["p0"]]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-coal-stack", ["p0"]]]},
    ),
    8: make_instance(
        8,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}},
        goals={"global": [["has-ironworks", ["p0"]], [">=", ["available", ["iron", "p0"]], "1"]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-coal-stack", ["p0"]]]},
    ),
    9: make_instance(
        9,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain", "metalliferous", "by-coast"}},
        goals={"global": [["has-docks", ["p0"]], [">=", ["housing", ["p0"]], "1"]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-coal-stack", ["p0"]]]},
    ),
    10: make_instance(
        10,
        places=["p0"],
        terrains={"p0": {"woodland", "mountain", "metalliferous", "by-coast"}},
        goals={"global": [["has-wharf", ["p0"]], [">=", ["housing", ["p0"]], "2"]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-coal-stack", ["p0"]]]},
    ),
    11: make_instance(
        11,
        places=["p0", "p1"],
        terrains={"p0": {"woodland", "mountain"}, "p1": {"woodland", "metalliferous"}},
        land_edges=[("p0", "p1")],
        goals={"global": [["has-quarry", ["p0"]], ["has-mine", ["p1"]]], "a0": [["has-cabin", ["p0"]]], "a1": [["has-cabin", ["p1"]]]},
    ),
    12: make_instance(
        12,
        places=["p0", "p1"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}, "p1": {"woodland", "mountain"}},
        land_edges=[("p0", "p1")],
        goals={"global": [["connected-by-rail", ["p0", "p1"]], [">=", ["housing", ["p0"]], "1"]], "a0": [["has-sawmill", ["p0"]]], "a1": [["has-cabin", ["p1"]]]},
    ),
    13: make_instance(
        13,
        places=["p0", "p1"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}, "p1": {"woodland", "mountain"}},
        land_edges=[("p0", "p1")],
        trains=["t0"],
        agent_init_extra={"a0": [["assigned-train", ["t0"]]]},
        goals={"global": [["connected-by-rail", ["p0", "p1"]]], "a0": [["owns_train", ["t0"]]], "a1": [["has-cabin", ["p1"]]]},
    ),
    14: make_instance(
        14,
        places=["p0", "p1"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}, "p1": {"woodland", "mountain", "metalliferous"}},
        land_edges=[("p0", "p1")],
        trains=["t0", "t1"],
        agent_init_extra={"a0": [["assigned-train", ["t0"]]], "a1": [["assigned-train", ["t1"]]]},
        goals={"global": [["connected-by-rail", ["p0", "p1"]]], "a0": [["owns_train", ["t0"]]], "a1": [["owns_train", ["t1"]]]},
    ),
    15: make_instance(
        15,
        places=["p0", "p1", "p2"],
        terrains={"p0": {"woodland", "mountain"}, "p1": {"woodland", "mountain", "metalliferous", "by-coast"}, "p2": {"woodland", "metalliferous"}},
        land_edges=[("p0", "p1"), ("p1", "p2")],
        goals={"global": [["has-docks", ["p1"]], ["connected-by-rail", ["p0", "p1"]], ["has-mine", ["p2"]]], "a0": [["has-cabin", ["p0"]]], "a1": [["has-sawmill", ["p2"]]]},
    ),
    16: make_instance(
        16,
        places=["p0", "p1", "p2"],
        terrains={"p0": {"woodland", "mountain"}, "p1": {"woodland", "mountain", "metalliferous", "by-coast"}, "p2": {"woodland", "metalliferous"}},
        land_edges=[("p0", "p1"), ("p1", "p2")],
        boats=["b0"],
        agent_init_extra={"a1": [["assigned-boat", ["b0"]]]},
        goals={"global": [["has-wharf", ["p1"]], ["connected-by-rail", ["p0", "p1"]]], "a0": [["has-cabin", ["p0"]]], "a1": [["owns_boat", ["b0"]]]},
    ),
    17: make_instance(
        17,
        places=["p0", "p1", "p2"],
        terrains={"p0": {"woodland", "mountain", "metalliferous", "by-coast"}, "p1": {"woodland", "mountain", "metalliferous"}, "p2": {"woodland", "mountain", "by-coast"}},
        land_edges=[("p0", "p1"), ("p1", "p2")],
        sea_edges=[("p0", "p2")],
        trains=["t0"],
        boats=["b0"],
        agent_init_extra={"a0": [["assigned-train", ["t0"]]], "a1": [["assigned-boat", ["b0"]]]},
        goals={"global": [["has-wharf", ["p0"]], ["connected-by-rail", ["p0", "p1"]], [">=", ["housing", ["p1"]], "1"]], "a0": [["owns_train", ["t0"]]], "a1": [["owns_boat", ["b0"]]]},
    ),
    18: make_instance(
        18,
        places=["p0", "p1", "p2", "p3"],
        terrains={"p0": {"woodland", "mountain", "metalliferous"}, "p1": {"woodland", "mountain"}, "p2": {"woodland", "mountain", "metalliferous", "by-coast"}, "p3": {"woodland", "metalliferous"}},
        land_edges=[("p0", "p1"), ("p1", "p2"), ("p2", "p3")],
        sea_edges=[("p0", "p2")],
        trains=["t0", "t1"],
        boats=["b0"],
        agent_init_extra={"a0": [["assigned-train", ["t0"]]], "a1": [["assigned-train", ["t1"]], ["assigned-boat", ["b0"]]]},
        goals={"global": [["has-wharf", ["p2"]], ["connected-by-rail", ["p0", "p1"]], ["connected-by-rail", ["p2", "p3"]], [">=", ["housing", ["p0"]], "1"], [">=", ["housing", ["p2"]], "1"]], "a0": [["owns_train", ["t0"]]], "a1": [["owns_train", ["t1"]], ["owns_boat", ["b0"]]]},
    ),
    19: make_instance(
        19,
        places=["p0", "p1", "p2", "p3"],
        terrains={"p0": {"woodland", "mountain", "metalliferous", "by-coast"}, "p1": {"woodland", "mountain", "metalliferous"}, "p2": {"woodland", "mountain", "metalliferous", "by-coast"}, "p3": {"woodland", "mountain"}},
        land_edges=[("p0", "p1"), ("p1", "p2"), ("p2", "p3")],
        sea_edges=[("p0", "p2")],
        trains=["t0", "t1"],
        boats=["b0", "b1"],
        agent_init_extra={"a0": [["assigned-train", ["t0"]], ["assigned-boat", ["b0"]]], "a1": [["assigned-train", ["t1"]], ["assigned-boat", ["b1"]]]},
        goals={"global": [["has-wharf", ["p0"]], ["has-wharf", ["p2"]], ["connected-by-rail", ["p0", "p1"]], ["connected-by-rail", ["p2", "p3"]], [">=", ["housing", ["p2"]], "2"]], "a0": [["owns_train", ["t0"]], ["owns_boat", ["b0"]]], "a1": [["owns_train", ["t1"]], ["owns_boat", ["b1"]]]},
    ),
    20: make_instance(
        20,
        places=["p0", "p1", "p2", "p3", "p4"],
        terrains={
            "p0": {"woodland", "mountain", "metalliferous", "by-coast"},
            "p1": {"woodland", "mountain"},
            "p2": {"woodland", "mountain", "metalliferous", "by-coast"},
            "p3": {"woodland", "metalliferous"},
            "p4": {"woodland", "mountain", "metalliferous", "by-coast"},
        },
        land_edges=[("p0", "p1"), ("p1", "p2"), ("p2", "p3"), ("p3", "p4")],
        sea_edges=[("p0", "p2"), ("p2", "p4")],
        trains=["t0", "t1"],
        boats=["b0", "b1"],
        agent_init_extra={"a0": [["assigned-train", ["t0"]], ["assigned-boat", ["b0"]]], "a1": [["assigned-train", ["t1"]], ["assigned-boat", ["b1"]]]},
        goals={
            "global": [
                ["has-wharf", ["p0"]],
                ["has-wharf", ["p4"]],
                ["connected-by-rail", ["p0", "p1"]],
                ["connected-by-rail", ["p2", "p3"]],
                [">=", ["housing", ["p2"]], "2"],
                [">=", ["housing", ["p4"]], "1"],
                ["has-mine", ["p3"]],
            ],
            "a0": [["owns_train", ["t0"]], ["owns_boat", ["b0"]]],
            "a1": [["owns_train", ["t1"]], ["owns_boat", ["b1"]]],
        },
    ),
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for idx, data in INSTANCES.items():
        path = ROOT / f"pfile{idx}.json"
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
