from experimentation.problem_generators.problem_generator import ProblemGenerator
import random

import unified_planning
from unified_planning.model import Fluent, InstantaneousAction, DurativeAction, StartTiming, EndTiming, \
    OpenTimeInterval, ClosedTimeInterval
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Dot

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw

class IntersectionProblemGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()

    def generate_problem(self, file_name=None, sl=False):
        cars=None
        yields_list=[]
        wait_drive=True
        durative=False

        if cars is None:
            cars = ["car-north", "car-south", "car-east", "car-west"]
        problem = MultiAgentProblemWithWaitfor("intersection")

        loc = UserType("loc")
        direction = UserType("direction")
        car = UserType("car")

        # Environment
        connected = Fluent('connected', BoolType(), l1=loc, l2=loc, d=direction)
        free = Fluent('free', BoolType(), l=loc)
        if len(yields_list) > 0:
            yields_to = Fluent('yields_to', BoolType(), l1=loc, l2=loc)
            problem.ma_environment.add_fluent(yields_to, default_initial_value=False)
            dummy_loc = unified_planning.model.Object("dummy", loc)
            problem.add_object(dummy_loc)

        problem.ma_environment.add_fluent(connected, default_initial_value=False)
        problem.ma_environment.add_fluent(free, default_initial_value=True)

        intersection_map = {
            "north": ["south-ent", "cross-se", "cross-ne", "north-ex"],
            "south": ["north-ent", "cross-nw", "cross-sw", "south-ex"],
            "west": ["east-ent", "cross-ne", "cross-nw", "west-ex"],
            "east": ["west-ent", "cross-sw", "cross-se", "east-ex"]
        }

        location_names = set()
        for l in intersection_map.values():
            location_names = location_names.union(l)
        locations = list(map(lambda l: unified_planning.model.Object(l, loc), location_names))
        problem.add_objects(locations)

        direction_names = intersection_map.keys()
        directions = list(map(lambda d: unified_planning.model.Object(d, direction), direction_names))
        problem.add_objects(directions)

        for d, l in intersection_map.items():
            for i in range(len(l) - 1):
                problem.set_initial_value(
                    connected(unified_planning.model.Object(l[i], loc), unified_planning.model.Object(l[i + 1], loc),
                              unified_planning.model.Object(d, direction)), True)

        # Agents
        at = Fluent('at', BoolType(), l1=loc)
        arrived = Fluent('arrived', BoolType())
        not_arrived = Fluent('not-arrived', BoolType())
        start = Fluent('start', BoolType(), l=loc)
        traveldirection = Fluent('traveldirection', BoolType(), d=direction)

        if durative:
            arrive = DurativeAction('arrive', l=loc)
            arrive.set_fixed_duration(1)
            l = arrive.parameter('l')

            arrive.add_condition(StartTiming(), start(l))
            arrive.add_condition(StartTiming(), not_arrived())
            arrive.add_condition(OpenTimeInterval(StartTiming(), EndTiming()), free(l))
            arrive.add_effect(EndTiming(), at(l), True)
            arrive.add_effect(EndTiming(), free(l), False)
            arrive.add_effect(EndTiming(), arrived(), True)
            arrive.add_effect(EndTiming(), not_arrived(), False)
        else:
            arrive = InstantaneousAction('arrive', l=loc)
            l = arrive.parameter('l')
            arrive.add_precondition(start(l))
            arrive.add_precondition(not_arrived())
            arrive.add_precondition(free(l))
            arrive.add_effect(at(l), True)
            arrive.add_effect(free(l), False)
            arrive.add_effect(arrived(), True)
            arrive.add_effect(not_arrived(), False)
        if durative:
            if len(yields_list) > 0:
                drive = DurativeAction('drive', l1=loc, l2=loc, d=direction, ly=loc)
            else:
                drive = DurativeAction('drive', l1=loc, l2=loc, d=direction)
            drive.set_fixed_duration(1)
            l1 = drive.parameter('l1')
            l2 = drive.parameter('l2')
            d = drive.parameter('d')
            drive.add_condition(StartTiming(), at(l1))
            if wait_drive:
                drive.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), free(l2))
            drive.add_condition(StartTiming(), traveldirection(d))
            drive.add_condition(EndTiming(), connected(l1, l2, d))
            drive.add_effect(EndTiming(), at(l2), True)
            drive.add_effect(EndTiming(), free(l2), False)
            drive.add_effect(StartTiming(), at(l1), False)
            drive.add_effect(EndTiming(), free(l1), True)
            if len(yields_list) > 0:
                ly = drive.parameter('ly')
                drive.add_condition(StartTiming(), yields_to(l1, ly))
                drive.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), free(ly))

        else:
            if len(yields_list) > 0:
                drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction, ly=loc)
            else:
                drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction)
            l1 = drive.parameter('l1')
            l2 = drive.parameter('l2')
            d = drive.parameter('d')
            # ly = drive.parameter('ly')
            drive.add_precondition(at(l1))
            drive.add_precondition(free(l2))  # Remove for yield/wait
            drive.add_precondition(traveldirection(d))
            drive.add_precondition(connected(l1, l2, d))
            if len(yields_list) > 0:
                ly = drive.parameter('ly')
                drive.add_precondition(yields_to(l1, ly))
                drive.add_precondition(free(ly))
            drive.add_effect(at(l2), True)
            drive.add_effect(free(l2), False)
            drive.add_effect(at(l1), False)

            drive.add_effect(free(l1), True)

        for d, l in intersection_map.items():
            carname = "car-" + d
            if carname in cars:
                car = Agent(carname, problem)

                problem.add_agent(car)
                car.add_fluent(at, default_initial_value=False)
                car.add_fluent(arrived, default_initial_value=False)
                car.add_fluent(not_arrived, default_initial_value=True)
                car.add_fluent(start, default_initial_value=False)
                car.add_fluent(traveldirection, default_initial_value=False)
                car.add_action(arrive)
                car.add_action(drive)

                slname = l[0]
                slobj = unified_planning.model.Object(slname, loc)

                glname = l[-1]
                globj = unified_planning.model.Object(glname, loc)

                dobj = unified_planning.model.Object(d, direction)

                problem.set_initial_value(Dot(car, car.fluent("start")(slobj)), True)
                problem.set_initial_value(Dot(car, car.fluent("traveldirection")(dobj)), True)
                car.add_public_goal(car.fluent("at")(globj))
                # problem.add_goal(Dot(car, car.fluent("at")(globj)))

                if len(yields_list) > 0:
                    yields = set()
                    for l1_name, ly_name in yields_list:
                        problem.set_initial_value(yields_to(problem.object(l1_name), problem.object(ly_name)), True)
                        yields.add(problem.object(l1_name))
                    for l1 in problem.objects(loc):
                        if l1 not in yields:
                            problem.set_initial_value(yields_to(l1, dummy_loc), True)

        # Add waitfor annotations
        for agent in problem.agents:
            drive = agent.action("drive")
            l2 = drive.parameter("l2")
            if wait_drive:
                problem.waitfor.annotate_as_waitfor(agent.name, drive.name, free(l2))
            if len(yields_list) > 0:
                ly = drive.parameter("ly")
                problem.waitfor.annotate_as_waitfor(agent.name, drive.name, free(ly))

        return problem

    def add_social_law_1(self):
        prob = self.problem
        l = SocialLaw()
        for agent in prob.agents:
            l.add_waitfor_annotation(agent.name, "drive", "free", ("l2",))

        res = l.compile(prob)
        return res.problem


    def add_social_law_3(self):
        p_4cars_deadlock = self.add_social_law()
        prob = self.problem
        l3 = SocialLaw()
        l3.add_new_fluent(None, "yieldsto", (("l1", "loc"), ("l2", "loc")), False)
        l3.add_new_object("dummy_loc", "loc")
        for loc1, loc2 in [("south-ent", "cross-ne"), ("north-ent", "cross-sw"), ("east-ent", "cross-nw"),
                           ("west-ent", "cross-se")]:
            l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc1, loc2), True)
        for loc in prob.objects(prob.user_type("loc")):
            if loc.name not in ["south-ent", "north-ent", "east-ent", "west-ent"]:
                l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc.name, "dummy_loc"), True)
        for agent in prob.agents:
            l3.add_parameter_to_action(agent.name, "drive", "ly", "loc")
            l3.add_precondition_to_action(agent.name, "drive", "yieldsto", ("l1", "ly"))
            l3.add_precondition_to_action(agent.name, "drive", "free", ("ly",))
            l3.add_waitfor_annotation(agent.name, "drive", "free", ("ly",))
        return l3.compile(p_4cars_deadlock).problem
