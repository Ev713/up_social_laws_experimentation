import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import BoolType, IntType, Not, LT, GT, Equals

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor


class NumericIntersectionGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()
        self.n = 1
        self.grid_size = 3
        self.vertical_cols = []
        self.horizontal_rows = []
        self.road_locs = []
        self.route_map = {}
        self.agent_lane = {}
        self.state = {}

    def location_name(self, row, col):
        return f"r{row}_c{col}"

    def _build_geometry(self):
        self.grid_size = (2 * self.n) + 1
        self.vertical_cols = list(range(1, self.grid_size, 2))
        self.horizontal_rows = list(range(1, self.grid_size, 2))

        road_locs = set()
        for col in self.vertical_cols:
            for row in range(self.grid_size):
                road_locs.add((row, col))
        for row in self.horizontal_rows:
            for col in range(self.grid_size):
                road_locs.add((row, col))
        self.road_locs = sorted(road_locs)

    def _build_routes(self):
        self.route_map = {}
        self.agent_lane = {}

        for lane_index, col in enumerate(self.vertical_cols, start=1):
            agent_name = f"car-south-{lane_index}"
            self.agent_lane[agent_name] = lane_index
            self.route_map[agent_name] = [self.location_name(row, col) for row in range(self.grid_size)]

        for lane_index, row in enumerate(self.horizontal_rows, start=1):
            agent_name = f"car-west-{lane_index}"
            self.agent_lane[agent_name] = lane_index
            self.route_map[agent_name] = [self.location_name(row, col) for col in range(self.grid_size - 1, -1, -1)]

    def _conflicting_south_agent(self, dst_name):
        row, col = (int(piece[1:]) for piece in dst_name.split("_"))
        if row not in self.horizontal_rows or col not in self.vertical_cols:
            return None, None
        lane_index = self.vertical_cols.index(col) + 1
        return f"car-south-{lane_index}", row

    def generate_problem(self, file_name=None, sl=False):
        if file_name is not None:
            self.load_instance_data(file_name)
            self.n = int(self.instance_data.get("n", 1))
        else:
            self.n = 1

        self._build_geometry()
        self._build_routes()
        self.state = {}

        self.problem = MultiAgentProblemWithWaitfor(f"numeric_intersection_n{self.n}")

        loc = unified_planning.shortcuts.UserType("loc")

        self.problem.add_objects(
            [unified_planning.model.Object(self.location_name(row, col), loc) for row, col in self.road_locs]
        )

        free = Fluent("free", BoolType(), l=loc)
        self.problem.ma_environment.add_fluent(free, default_initial_value=True)

        for agent_name in self.route_map:
            agent = Agent(agent_name, self.problem)
            self.problem.add_agent(agent)

            at = Fluent(f"{agent_name}_at", BoolType(), l=loc)
            on_map = Fluent(f"{agent_name}_on_map", BoolType())
            progress = Fluent(f"{agent_name}_progress", IntType(0, self.grid_size - 1))
            start = Fluent(f"{agent_name}_start", BoolType(), l=loc)
            self.state[agent_name] = {
                "at": at,
                "on_map": on_map,
                "progress": progress,
                "start": start,
            }
            self.problem.ma_environment.add_fluent(at, default_initial_value=False)
            self.problem.ma_environment.add_fluent(on_map, default_initial_value=False)
            self.problem.ma_environment.add_fluent(progress, default_initial_value=0)
            self.problem.ma_environment.add_fluent(start, default_initial_value=False)

            route = self.route_map[agent_name]
            start_obj = self.problem.object(route[0])
            goal_obj = self.problem.object(route[-1])

            self.problem.set_initial_value(start(start_obj), True)
            self.problem.set_initial_value(progress(), 0)

            appear = InstantaneousAction("appear", l=loc)
            l = appear.parameter("l")
            appear.add_precondition(start(l))
            appear.add_precondition(Not(on_map()))
            appear.add_precondition(free(l))
            appear.add_effect(on_map(), True)
            appear.add_effect(at(l), True)
            appear.add_effect(free(l), False)
            agent.add_action(appear)
            self.problem.waitfor.annotate_as_waitfor(agent.name, appear.name, free(l))

            for step in range(len(route) - 1):
                src_name = route[step]
                dst_name = route[step + 1]
                src_obj = self.problem.object(src_name)
                dst_obj = self.problem.object(dst_name)

                move_names = [f"move_{step}"]
                move_preconditions = [[]]
                if sl and agent_name.startswith("car-west-"):
                    conflicting_agent, conflict_row = self._conflicting_south_agent(dst_name)
                    if conflicting_agent is not None:
                        conflicting_progress = self.state[conflicting_agent]["progress"]
                        conflicting_on_map = self.state[conflicting_agent]["on_map"]
                        move_names = [f"move_before_{step}", f"move_after_{step}"]
                        move_preconditions = [
                            [Not(conflicting_on_map())],
                            [GT(conflicting_progress(), conflict_row)],
                        ]

                for action_name, extra_preconditions in zip(move_names, move_preconditions):
                    move = InstantaneousAction(action_name)
                    move.add_precondition(on_map())
                    move.add_precondition(at(src_obj))
                    move.add_precondition(free(dst_obj))
                    move.add_precondition(Equals(progress(), step))
                    for prec in extra_preconditions:
                        move.add_precondition(prec)
                    move.add_effect(at(src_obj), False)
                    move.add_effect(at(dst_obj), True)
                    move.add_effect(free(src_obj), True)
                    move.add_effect(free(dst_obj), False)
                    move.add_increase_effect(progress(), 1)
                    agent.add_action(move)
                    self.problem.waitfor.annotate_as_waitfor(agent.name, move.name, free(dst_obj))

            agent.add_public_goal(at(goal_obj))

        return self.problem
