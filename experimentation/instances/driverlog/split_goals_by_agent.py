#!/usr/bin/env python3
"""Rewrite Driverlog instance goals as explicit per-agent goal maps.

Truck location goals are assigned to the agent that owns that truck by default
(agent/truck zip order). Non-truck goals are assigned round-robin across agents.
"""

import json
from pathlib import Path


def split_goals(data):
    goals = data.get("goals", [])
    agents = data["agents"]
    trucks = data["trucks"]
    truck_owner = dict(zip(trucks, agents))
    agent_goals = {agent: [] for agent in agents}

    if isinstance(goals, dict):
        for agent in agents:
            for goal in goals.get(agent, []):
                agent_goals[agent].append(goal)
        return agent_goals

    round_robin_index = 0
    for goal in goals:
        if goal[0] == "at" and goal[1][0] in truck_owner:
            agent_goals[truck_owner[goal[1][0]]].append(goal)
            continue
        agent = agents[round_robin_index % len(agents)]
        agent_goals[agent].append(goal)
        round_robin_index += 1
    return agent_goals


def main():
    base = Path(__file__).resolve().parent
    for path in sorted(base.glob("pfile*.json")):
        data = json.loads(path.read_text())
        data["goals"] = split_goals(data)
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(path.name)


if __name__ == "__main__":
    main()
