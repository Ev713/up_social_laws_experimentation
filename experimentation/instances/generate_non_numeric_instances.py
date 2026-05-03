import json
import shutil
from ast import literal_eval
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LEGACY_JSON = ROOT.parent / "legacy" / "problem_instances" / "all" / "jsons"


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def populate_driverlog():
    src = LEGACY_JSON / "driverlog"
    dst = ROOT / "driverlog"
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("pfile*.json"), key=lambda p: int(p.stem[5:])):
        shutil.copy2(path, dst / path.name)


def populate_zenotravel():
    src = LEGACY_JSON / "zenotravel"
    dst = ROOT / "zenotravel"
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("pfile*.json"), key=lambda p: int(p.stem[5:]))
    for i, path in enumerate(paths, start=1):
        shutil.copy2(path, dst / f"pfile{i}.json")


def populate_blocksworld():
    src = LEGACY_JSON / "blocksworld"
    dst = ROOT / "blocksworld"
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*.json"))
    for i, path in enumerate(paths, start=1):
        shutil.copy2(path, dst / f"pfile{i}.json")


GRID_INITS = [
    {"car-0": "(1, 1)", "car-1": "(1, 1)"},
    {"car-0": "(2, 0)", "car-1": "(0, 2)", "car-2": "(1, 0)"},
    {"car-0": "(0, 2)", "car-1": "(1, 1)", "car-2": "(2, 3)", "car-3": "(1, 2)"},
    {"car-0": "(0, 2)", "car-1": "(1, 3)", "car-2": "(3, 0)", "car-3": "(2, 1)", "car-4": "(0, 3)"},
    {"car-0": "(3, 2)", "car-1": "(3, 1)", "car-2": "(0, 3)", "car-3": "(3, 2)", "car-4": "(0, 0)", "car-5": "(0, 3)"},
    {"car-0": "(1, 1)", "car-1": "(0, 0)", "car-2": "(2, 2)", "car-3": "(1, 3)", "car-4": "(3, 4)", "car-5": "(3, 0)"},
    {"car-0": "(3, 1)", "car-1": "(2, 4)", "car-2": "(4, 1)", "car-3": "(2, 0)", "car-4": "(3, 0)", "car-5": "(1, 4)", "car-6": "(3, 0)"},
    {"car-0": "(2, 5)", "car-1": "(1, 0)", "car-2": "(5, 4)", "car-3": "(5, 0)", "car-4": "(0, 4)", "car-5": "(1, 5)", "car-6": "(2, 1)", "car-7": "(0, 4)"},
    {"car-0": "(5, 3)", "car-1": "(1, 2)", "car-2": "(1, 3)", "car-3": "(2, 2)", "car-4": "(1, 6)", "car-5": "(1, 1)", "car-6": "(4, 3)", "car-7": "(4, 1)"},
    {"car-0": "(3, 0)", "car-1": "(4, 2)", "car-2": "(1, 1)", "car-3": "(6, 6)", "car-4": "(5, 5)", "car-5": "(4, 0)", "car-6": "(5, 6)"},
    {"car-0": "(2, 7)", "car-1": "(5, 5)", "car-2": "(6, 0)", "car-3": "(1, 3)", "car-4": "(0, 7)", "car-5": "(0, 3)", "car-6": "(4, 5)", "car-7": "(1, 2)", "car-8": "(0, 1)"},
    {"car-0": "(6, 1)", "car-1": "(3, 0)"},
    {"car-0": "(2, 2)", "car-1": "(5, 2)", "car-2": "(2, 0)"},
    {"car-0": "(4, 1)", "car-1": "(2, 0)", "car-2": "(0, 1)", "car-3": "(2, 3)", "car-4": "(0, 1)", "car-5": "(3, 2)"},
    {"car-0": "(0, 1)", "car-1": "(6, 1)", "car-2": "(0, 0)", "car-3": "(0, 4)", "car-4": "(5, 2)", "car-5": "(2, 2)", "car-6": "(5, 1)"},
    {"car-0": "(0, 4)", "car-1": "(7, 2)", "car-2": "(3, 2)", "car-3": "(4, 2)", "car-4": "(0, 3)", "car-5": "(5, 1)", "car-6": "(7, 3)", "car-7": "(6, 4)"},
    {"car-0": "(6, 6)", "car-1": "(4, 1)", "car-2": "(7, 1)", "car-3": "(5, 6)", "car-4": "(0, 4)", "car-5": "(1, 0)", "car-6": "(1, 1)", "car-7": "(2, 4)", "car-8": "(1, 1)", "car-9": "(5, 0)"},
    {"car-0": "(3, 3)", "car-1": "(3, 4)", "car-2": "(5, 3)", "car-3": "(4, 2)", "car-4": "(2, 7)", "car-5": "(1, 7)", "car-6": "(5, 4)", "car-7": "(1, 1)"},
    {"car-0": "(4, 6)", "car-1": "(2, 4)", "car-2": "(2, 5)", "car-3": "(2, 6)", "car-4": "(3, 1)", "car-5": "(4, 6)", "car-6": "(3, 7)", "car-7": "(4, 1)"},
    {"car-0": "(1, 4)", "car-1": "(0, 1)", "car-2": "(1, 1)"},
]

GRID_GOALS = [
    {"car-0": "(1, 2)", "car-1": "(0, 1)"},
    {"car-0": "(0, 2)", "car-1": "(2, 0)", "car-2": "(2, 2)"},
    {"car-0": "(1, 3)", "car-1": "(1, 3)", "car-2": "(2, 0)", "car-3": "(0, 0)"},
    {"car-0": "(2, 3)", "car-1": "(2, 3)", "car-2": "(3, 0)", "car-3": "(2, 1)", "car-4": "(3, 3)"},
    {"car-0": "(0, 4)", "car-1": "(3, 4)", "car-2": "(3, 2)", "car-3": "(2, 3)", "car-4": "(0, 1)", "car-5": "(1, 3)"},
    {"car-0": "(1, 1)", "car-1": "(3, 3)", "car-2": "(2, 4)", "car-3": "(4, 1)", "car-4": "(4, 2)", "car-5": "(0, 2)"},
    {"car-0": "(4, 2)", "car-1": "(3, 1)", "car-2": "(3, 1)", "car-3": "(3, 1)", "car-4": "(1, 1)", "car-5": "(0, 0)", "car-6": "(2, 1)"},
    {"car-0": "(1, 3)", "car-1": "(3, 1)", "car-2": "(5, 3)", "car-3": "(0, 1)", "car-4": "(4, 5)", "car-5": "(5, 0)", "car-6": "(1, 1)", "car-7": "(4, 2)"},
    {"car-0": "(3, 3)", "car-1": "(5, 2)", "car-2": "(3, 4)", "car-3": "(4, 1)", "car-4": "(3, 3)", "car-5": "(3, 6)", "car-6": "(1, 0)", "car-7": "(5, 4)"},
    {"car-0": "(4, 3)", "car-1": "(5, 2)", "car-2": "(0, 2)", "car-3": "(1, 0)", "car-4": "(1, 5)", "car-5": "(6, 4)", "car-6": "(2, 1)"},
    {"car-0": "(3, 1)", "car-1": "(4, 4)", "car-2": "(3, 3)", "car-3": "(1, 7)", "car-4": "(3, 2)", "car-5": "(1, 3)", "car-6": "(6, 2)", "car-7": "(6, 4)", "car-8": "(1, 7)"},
    {"car-0": "(1, 1)", "car-1": "(5, 0)"},
    {"car-0": "(4, 2)", "car-1": "(0, 0)", "car-2": "(5, 0)"},
    {"car-0": "(7, 1)", "car-1": "(5, 3)", "car-2": "(4, 1)", "car-3": "(2, 1)", "car-4": "(6, 2)", "car-5": "(0, 1)"},
    {"car-0": "(6, 0)", "car-1": "(4, 4)", "car-2": "(7, 2)", "car-3": "(0, 3)", "car-4": "(3, 1)", "car-5": "(5, 0)", "car-6": "(2, 2)"},
    {"car-0": "(5, 5)", "car-1": "(5, 5)", "car-2": "(7, 3)", "car-3": "(1, 3)", "car-4": "(5, 2)", "car-5": "(6, 1)", "car-6": "(6, 4)", "car-7": "(0, 0)"},
    {"car-0": "(5, 2)", "car-1": "(0, 2)", "car-2": "(7, 6)", "car-3": "(1, 5)", "car-4": "(6, 5)", "car-5": "(0, 3)", "car-6": "(4, 1)", "car-7": "(7, 5)", "car-8": "(7, 1)", "car-9": "(0, 6)"},
    {"car-0": "(0, 2)", "car-1": "(0, 2)", "car-2": "(4, 7)", "car-3": "(5, 6)", "car-4": "(4, 0)", "car-5": "(5, 5)", "car-6": "(0, 0)", "car-7": "(5, 5)"},
    {"car-0": "(4, 5)", "car-1": "(3, 4)", "car-2": "(2, 6)", "car-3": "(2, 3)", "car-4": "(2, 3)", "car-5": "(2, 1)", "car-6": "(2, 5)", "car-7": "(0, 5)"},
    {"car-0": "(1, 4)", "car-1": "(2, 3)", "car-2": "(2, 3)"},
]


def populate_grid():
    dst = ROOT / "grid"
    dst.mkdir(parents=True, exist_ok=True)
    for i, (inits, goals) in enumerate(zip(GRID_INITS, GRID_GOALS), start=1):
        coords = [literal_eval(loc) for loc in list(inits.values()) + list(goals.values())]
        width = max(c[0] for c in coords) + 1
        height = max(c[1] for c in coords) + 1
        data = {
            "width": width,
            "height": height,
            "agents": sorted(inits.keys(), key=lambda x: int(x.split("-")[-1])),
            "init_locs": inits,
            "goal_locs": goals,
        }
        write_json(dst / f"pfile{i}.json", data)


def populate_intersection():
    dst = ROOT / "intersection"
    dst.mkdir(parents=True, exist_ok=True)
    templates = [
        {"cars": ["car-north", "car-south"], "yields_list": [], "wait_drive": True, "durative": False},
        {"cars": ["car-east", "car-west"], "yields_list": [], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-east", "car-west"], "yields_list": [], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-south", "car-east", "car-west"], "yields_list": [], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-south", "car-east", "car-west"], "yields_list": [["south-ent", "cross-ne"]], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-south", "car-east", "car-west"], "yields_list": [["south-ent", "cross-ne"], ["north-ent", "cross-sw"]], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-south", "car-east", "car-west"], "yields_list": [["south-ent", "cross-ne"], ["north-ent", "cross-sw"], ["east-ent", "cross-nw"]], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-south", "car-east", "car-west"], "yields_list": [["south-ent", "cross-ne"], ["north-ent", "cross-sw"], ["east-ent", "cross-nw"], ["west-ent", "cross-se"]], "wait_drive": True, "durative": False},
        {"cars": ["car-north", "car-south", "car-east"], "yields_list": [["south-ent", "cross-ne"]], "wait_drive": True, "durative": True},
        {"cars": ["car-north", "car-south", "car-east", "car-west"], "yields_list": [["south-ent", "cross-ne"]], "wait_drive": True, "durative": True},
    ]
    for i in range(20):
        template = templates[min(i, len(templates) - 1)]
        write_json(dst / f"pfile{i + 1}.json", template)


def main():
    populate_driverlog()
    populate_zenotravel()
    populate_blocksworld()
    populate_grid()
    populate_intersection()


if __name__ == "__main__":
    main()
