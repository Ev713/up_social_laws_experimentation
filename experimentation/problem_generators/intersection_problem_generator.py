import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Dot

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


class IntersectionProblemGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()
        self.n = 1
        self.grid_size = 3
        self.road_locs = set()
        self.vertical_cols = []
        self.horizontal_rows = []

    def location_name(self, row, col):
        return f"r{row}_c{col}"

    def lane_coordinates(self):
        self.grid_size = (2 * self.n) + 1
        self.vertical_cols = list(range(1, self.grid_size, 2))
        self.horizontal_rows = list(range(1, self.grid_size, 2))
        self.road_locs = set()

        for col in self.vertical_cols:
            for row in range(self.grid_size):
                self.road_locs.add((row, col))

        for row in self.horizontal_rows:
            for col in range(self.grid_size):
                self.road_locs.add((row, col))

    def is_intersection(self, row, col):
        return row in self.horizontal_rows and col in self.vertical_cols

    def is_west_agent(self, agent_name):
        return agent_name.startswith("car-west-")

    def social_law_yields(self):
        dummy_name = "dummy_loc"
        law = SocialLaw()
        law.add_new_fluent(None, "yields_to", (("l1", "loc"), ("l2", "loc")), False)
        law.add_new_object(dummy_name, "loc")

        for agent in self.problem.agents:
            law.add_waitfor_annotation(agent.name, "drive", "free", ("l2",))

            if not self.is_west_agent(agent.name):
                continue

            law.add_parameter_to_action(agent.name, "drive", "ly", "loc")
            law.add_precondition_to_action(agent.name, "drive", "yields_to", ("l1", "ly"))
            law.add_precondition_to_action(agent.name, "drive", "free", ("ly",))
            law.add_waitfor_annotation(agent.name, "drive", "free", ("ly",))

        yield_sources = set()
        for row in self.horizontal_rows:
            for col in self.vertical_cols:
                source = self.location_name(row, col + 1)
                north_of_intersection = self.location_name(row - 1, col)
                law.set_initial_value_for_new_fluent(None, "yields_to", (source, north_of_intersection), True)
                yield_sources.add(source)

        for row, col in sorted(self.road_locs):
            source = self.location_name(row, col)
            if source in yield_sources:
                continue
            law.set_initial_value_for_new_fluent(None, "yields_to", (source, dummy_name), True)

        return law.compile(self.problem).problem

    def generate_problem(self, file_name=None, sl=False):
        if file_name is not None:
            self.load_instance_data(file_name)
            self.n = int(self.instance_data.get("n", 1))
            problem_name = f"intersection_n{self.n}"
        else:
            self.n = 1
            problem_name = "intersection_n1"

        self.lane_coordinates()
        self.problem = MultiAgentProblemWithWaitfor(problem_name)

        loc = UserType("loc")
        direction = UserType("direction")

        connected = Fluent("connected", BoolType(), l1=loc, l2=loc, d=direction)
        free = Fluent("free", BoolType(), l=loc)
        self.problem.ma_environment.add_fluent(connected, default_initial_value=False)
        self.problem.ma_environment.add_fluent(free, default_initial_value=True)

        self.problem.add_objects(
            [unified_planning.model.Object(self.location_name(row, col), loc) for row, col in sorted(self.road_locs)]
        )
        self.problem.add_objects(
            [
                unified_planning.model.Object("south", direction),
                unified_planning.model.Object("west", direction),
            ]
        )

        for col in self.vertical_cols:
            for row in range(self.grid_size - 1):
                self.problem.set_initial_value(
                    connected(
                        self.problem.object(self.location_name(row, col)),
                        self.problem.object(self.location_name(row + 1, col)),
                        self.problem.object("south"),
                    ),
                    True,
                )

        for row in self.horizontal_rows:
            for col in range(1, self.grid_size):
                self.problem.set_initial_value(
                    connected(
                        self.problem.object(self.location_name(row, col)),
                        self.problem.object(self.location_name(row, col - 1)),
                        self.problem.object("west"),
                    ),
                    True,
                )

        at = Fluent("at", BoolType(), l=loc)
        not_arrived = Fluent("not_arrived", BoolType())
        start = Fluent("start", BoolType(), l=loc)
        travel_direction = Fluent("travel_direction", BoolType(), d=direction)

        arrive = InstantaneousAction("arrive", l=loc)
        l = arrive.parameter("l")
        arrive.add_precondition(start(l))
        arrive.add_precondition(not_arrived())
        arrive.add_precondition(free(l))
        arrive.add_effect(at(l), True)
        arrive.add_effect(free(l), False)
        arrive.add_effect(not_arrived(), False)

        drive = InstantaneousAction("drive", l1=loc, l2=loc, d=direction)
        l1 = drive.parameter("l1")
        l2 = drive.parameter("l2")
        d = drive.parameter("d")
        drive.add_precondition(at(l1))
        drive.add_precondition(free(l2))
        drive.add_precondition(connected(l1, l2, d))
        drive.add_precondition(travel_direction(d))
        drive.add_effect(at(l2), True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(l1), False)
        drive.add_effect(free(l1), True)

        for lane_index, col in enumerate(self.vertical_cols, start=1):
            agent_name = f"car-south-{lane_index}"
            agent = Agent(agent_name, self.problem)
            self.problem.add_agent(agent)
            agent.add_fluent(at, default_initial_value=False)
            agent.add_fluent(not_arrived, default_initial_value=True)
            agent.add_fluent(start, default_initial_value=False)
            agent.add_fluent(travel_direction, default_initial_value=False)
            agent.add_action(arrive)
            agent.add_action(drive)

            start_obj = self.problem.object(self.location_name(0, col))
            goal_obj = self.problem.object(self.location_name(self.grid_size - 1, col))
            south_obj = self.problem.object("south")

            self.problem.set_initial_value(Dot(agent, agent.fluent("start")(start_obj)), True)
            self.problem.set_initial_value(Dot(agent, agent.fluent("travel_direction")(south_obj)), True)
            agent.add_public_goal(agent.fluent("at")(goal_obj))

        for lane_index, row in enumerate(self.horizontal_rows, start=1):
            agent_name = f"car-west-{lane_index}"
            agent = Agent(agent_name, self.problem)
            self.problem.add_agent(agent)
            agent.add_fluent(at, default_initial_value=False)
            agent.add_fluent(not_arrived, default_initial_value=True)
            agent.add_fluent(start, default_initial_value=False)
            agent.add_fluent(travel_direction, default_initial_value=False)
            agent.add_action(arrive)
            agent.add_action(drive)

            start_obj = self.problem.object(self.location_name(row, self.grid_size - 1))
            goal_obj = self.problem.object(self.location_name(row, 0))
            west_obj = self.problem.object("west")

            self.problem.set_initial_value(Dot(agent, agent.fluent("start")(start_obj)), True)
            self.problem.set_initial_value(Dot(agent, agent.fluent("travel_direction")(west_obj)), True)
            agent.add_public_goal(agent.fluent("at")(goal_obj))

        if sl:
            self.problem = self.social_law_yields()
        return self.problem
