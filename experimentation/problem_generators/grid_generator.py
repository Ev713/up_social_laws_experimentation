import random

import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Dot

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


class GridGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()
        self.GOAL_LOCS = None
        self.INIT_LOCS = None
        self.directions = None
        self.compass = None
        self.intersections = None
        self.init_locs = None
        self.goal_locs = None
        self.width = None
        self.height = None
        self.num_of_agents = None

    def get_dir(self, l1, l2):
        diff = (l2[0] - l1[0], l2[1] - l1[1])
        return self.compass[diff]

    def set_parameters(self, width, height, agents):
        self.width = width
        self.height = height
        self.num_of_agents = agents
        self.init_locs = {}  # {a: (0, 0) for a in range(self.num_of_agents)}
        self.goal_locs = {}  # {a: (self.width - 1, self.height - 1) for a in range(self.num_of_agents)}
        self.intersections = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.compass = {(0, 1): 'north', (1, 0): 'east', (0, -1): 'south', (-1, 0): 'west'}
        self.directions = {str((l1, l2)): self.get_dir(l1, l2) for l1 in self.intersections for l2 in self.intersections
                           if (l2[0] - l1[0], l2[1] - l1[1]) in self.compass}

    def grid_instance_data(self):
        self.INIT_LOCS = [{0: '(1, 1)', 1: '(1, 1)'}, {0: '(2, 0)', 1: '(0, 2)', 2: '(1, 0)'},
                          {0: '(0, 2)', 1: '(1, 1)', 2: '(2, 3)', 3: '(1, 2)'},
                          {0: '(0, 2)', 1: '(1, 3)', 2: '(3, 0)', 3: '(2, 1)', 4: '(0, 3)'},
                          {0: '(3, 2)', 1: '(3, 1)', 2: '(0, 3)', 3: '(3, 2)', 4: '(0, 0)', 5: '(0, 3)'},
                          {0: '(1, 1)', 1: '(0, 0)', 2: '(2, 2)', 3: '(1, 3)', 4: '(3, 4)', 5: '(3, 0)'},
                          {0: '(3, 1)', 1: '(2, 4)', 2: '(4, 1)', 3: '(2, 0)', 4: '(3, 0)', 5: '(1, 4)', 6: '(3, 0)'},
                          {0: '(2, 5)', 1: '(1, 0)', 2: '(5, 4)', 3: '(5, 0)', 4: '(0, 4)', 5: '(1, 5)', 6: '(2, 1)',
                           7: '(0, 4)'},
                          {0: '(5, 3)', 1: '(1, 2)', 2: '(1, 3)', 3: '(2, 2)', 4: '(1, 6)', 5: '(1, 1)', 6: '(4, 3)',
                           7: '(4, 1)'},
                          {0: '(3, 0)', 1: '(4, 2)', 2: '(1, 1)', 3: '(6, 6)', 4: '(5, 5)', 5: '(4, 0)', 6: '(5, 6)'},
                          {0: '(2, 7)', 1: '(5, 5)', 2: '(6, 0)', 3: '(1, 3)', 4: '(0, 7)', 5: '(0, 3)', 6: '(4, 5)',
                           7: '(1, 2)',
                           8: '(0, 1)'}, {0: '(6, 1)', 1: '(3, 0)'}, {0: '(2, 2)', 1: '(5, 2)', 2: '(2, 0)'},
                          {0: '(4, 1)', 1: '(2, 0)', 2: '(0, 1)', 3: '(2, 3)', 4: '(0, 1)', 5: '(3, 2)'},
                          {0: '(0, 1)', 1: '(6, 1)', 2: '(0, 0)', 3: '(0, 4)', 4: '(5, 2)', 5: '(2, 2)', 6: '(5, 1)'},
                          {0: '(0, 4)', 1: '(7, 2)', 2: '(3, 2)', 3: '(4, 2)', 4: '(0, 3)', 5: '(5, 1)', 6: '(7, 3)',
                           7: '(6, 4)'},
                          {0: '(6, 6)', 1: '(4, 1)', 2: '(7, 1)', 3: '(5, 6)', 4: '(0, 4)', 5: '(1, 0)', 6: '(1, 1)',
                           7: '(2, 4)',
                           8: '(1, 1)', 9: '(5, 0)'},
                          {0: '(3, 3)', 1: '(3, 4)', 2: '(5, 3)', 3: '(4, 2)', 4: '(2, 7)', 5: '(1, 7)', 6: '(5, 4)',
                           7: '(1, 1)'},
                          {0: '(4, 6)', 1: '(2, 4)', 2: '(2, 5)', 3: '(2, 6)', 4: '(3, 1)', 5: '(4, 6)', 6: '(3, 7)',
                           7: '(4, 1)'},
                          {0: '(1, 4)', 1: '(0, 1)', 2: '(1, 1)'}]

        self.GOAL_LOCS = [{0: '(1, 2)', 1: '(0, 1)'}, {0: '(0, 2)', 1: '(2, 0)', 2: '(2, 2)'},
                          {0: '(1, 3)', 1: '(1, 3)', 2: '(2, 0)', 3: '(0, 0)'},
                          {0: '(2, 3)', 1: '(2, 3)', 2: '(3, 0)', 3: '(2, 1)', 4: '(3, 3)'},
                          {0: '(0, 4)', 1: '(3, 4)', 2: '(3, 2)', 3: '(2, 3)', 4: '(0, 1)', 5: '(1, 3)'},
                          {0: '(1, 1)', 1: '(3, 3)', 2: '(2, 4)', 3: '(4, 1)', 4: '(4, 2)', 5: '(0, 2)'},
                          {0: '(4, 2)', 1: '(3, 1)', 2: '(3, 1)', 3: '(3, 1)', 4: '(1, 1)', 5: '(0, 0)', 6: '(2, 1)'},
                          {0: '(1, 3)', 1: '(3, 1)', 2: '(5, 3)', 3: '(0, 1)', 4: '(4, 5)', 5: '(5, 0)', 6: '(1, 1)',
                           7: '(4, 2)'},
                          {0: '(3, 3)', 1: '(5, 2)', 2: '(3, 4)', 3: '(4, 1)', 4: '(3, 3)', 5: '(3, 6)', 6: '(1, 0)',
                           7: '(5, 4)'},
                          {0: '(4, 3)', 1: '(5, 2)', 2: '(0, 2)', 3: '(1, 0)', 4: '(1, 5)', 5: '(6, 4)', 6: '(2, 1)'},
                          {0: '(3, 1)', 1: '(4, 4)', 2: '(3, 3)', 3: '(1, 7)', 4: '(3, 2)', 5: '(1, 3)', 6: '(6, 2)',
                           7: '(6, 4)',
                           8: '(1, 7)'}, {0: '(1, 1)', 1: '(5, 0)'}, {0: '(4, 2)', 1: '(0, 0)', 2: '(5, 0)'},
                          {0: '(7, 1)', 1: '(5, 3)', 2: '(4, 1)', 3: '(2, 1)', 4: '(6, 2)', 5: '(0, 1)'},
                          {0: '(6, 0)', 1: '(4, 4)', 2: '(7, 2)', 3: '(0, 3)', 4: '(3, 1)', 5: '(5, 0)', 6: '(2, 2)'},
                          {0: '(5, 5)', 1: '(5, 5)', 2: '(7, 3)', 3: '(1, 3)', 4: '(5, 2)', 5: '(6, 1)', 6: '(6, 4)',
                           7: '(0, 0)'},
                          {0: '(5, 2)', 1: '(0, 2)', 2: '(7, 6)', 3: '(1, 5)', 4: '(6, 5)', 5: '(0, 3)', 6: '(4, 1)',
                           7: '(7, 5)',
                           8: '(7, 1)', 9: '(0, 6)'},
                          {0: '(0, 2)', 1: '(0, 2)', 2: '(4, 7)', 3: '(5, 6)', 4: '(4, 0)', 5: '(5, 5)', 6: '(0, 0)',
                           7: '(5, 5)'},
                          {0: '(4, 5)', 1: '(3, 4)', 2: '(2, 6)', 3: '(2, 3)', 4: '(2, 3)', 5: '(2, 1)', 6: '(2, 5)',
                           7: '(0, 5)'},
                          {0: '(1, 4)', 1: '(2, 3)', 2: '(2, 3)'}]

    def get_goal_loc(self, agent):
        if agent in self.goal_locs and self.goal_locs[agent] in self.intersections:
            return str(self.goal_locs[agent])
        else:
            self.goal_locs[agent] = str(random.choice(self.intersections))
            return self.goal_locs[agent]

    def get_init_loc(self, agent):
        if agent in self.init_locs and self.init_locs[agent] in self.intersections:
            return str(self.init_locs[agent])
        else:
            self.init_locs[agent] = str(random.choice(self.intersections))
            return self.init_locs[agent]

    def add_social_law(self):
        up_columns = []
        down_columns = []
        l = 0
        r = self.width - 1
        flag = True
        while True:
            if flag:
                up_columns.append(l)
                down_columns.append(r)
            else:
                up_columns.append(r)
                down_columns.append(l)
            flag = not flag
            l += 1
            r -= 1
            if l == r:
                up_columns.append(l)
                break
            if l > r:
                break
        direction_law = SocialLaw()
        for l1 in self.intersections:
            for l2 in self.intersections:
                if (l2[0] - l1[0], l2[1] - l1[1]) not in self.compass:
                    continue
                if self.get_dir(l1, l2) == 'north' and l1[0] in up_columns:
                    continue
                if self.get_dir(l1, l2) == 'south' and l1[0] in down_columns:
                    continue
                if self.get_dir(l1, l2) == 'east' and l1[1] == self.height - 1:
                    continue
                if self.get_dir(l1, l2) == 'west' and l1[1] == 0:
                    continue
                for a in range(self.num_of_agents):
                    carname = "car-" + str(a)
                    direction_law.disallow_action(carname, 'drive', (str(l1), str(l2), self.get_dir(l1, l2)))
        for a in range(self.num_of_agents):
            carname = "car-" + str(a)
            direction_law.add_waitfor_annotation(carname, 'drive', 'free', ('l2',))
        for a in range(self.num_of_agents):
            carname = "car-" + str(a)
            direction_law.add_waitfor_annotation(carname, 'arrive', 'free', ('l',))

        return direction_law.compile(self.problem).problem

    def generate_problem(self, file_name=None, sl=False):
        self.problem = MultiAgentProblemWithWaitfor()
        loc = UserType("loc")

        direction = UserType("direction")
        connected = Fluent('connected', BoolType(), l1=loc, l2=loc, d=direction)
        free = Fluent('free', BoolType(), l=loc)
        self.problem.ma_environment.add_fluent(connected, default_initial_value=False)
        self.problem.ma_environment.add_fluent(free, default_initial_value=True)
        self.problem.add_objects(list(map(lambda d: unified_planning.model.Object(d, direction),
                                          list(self.compass.values()))))
        self.problem.add_objects(
            list(map(lambda l: unified_planning.model.Object(l, loc), [str(x) for x in self.intersections])))

        for l1 in self.intersections:
            for l2 in self.intersections:
                if (l2[0] - l1[0], l2[1] - l1[1]) not in self.compass:
                    continue
                self.problem.set_initial_value(
                    connected(unified_planning.model.Object(str(l1), loc), unified_planning.model.Object(str(l2), loc),
                              unified_planning.model.Object(self.directions[str((l1, l2))], direction)), True)

        at = Fluent('at', BoolType(), l1=loc)
        left = Fluent('left', BoolType())
        not_arrived = Fluent('not-arrived', BoolType())
        start = Fluent('start', BoolType(), l=loc)
        goal = Fluent('goal', BoolType(), l=loc)

        # Arrive action
        arrive = InstantaneousAction('arrive', l=loc)
        l = arrive.parameter('l')
        arrive.add_precondition(start(l))
        arrive.add_precondition(not_arrived())
        arrive.add_precondition(free(l))
        arrive.add_effect(at(l), True)
        arrive.add_effect(free(l), False)
        arrive.add_effect(not_arrived(), False)

        # Drive action
        drive = InstantaneousAction('drive', l1=loc, l2=loc, d=direction)
        l1 = drive.parameter('l1')
        l2 = drive.parameter('l2')
        d = drive.parameter('d')
        drive.add_precondition(at(l1))
        drive.add_precondition(free(l2))
        drive.add_precondition(connected(l1, l2, d))
        drive.add_effect(at(l2), True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(l1), False)
        drive.add_effect(free(l1), True)

        # leave action
        leave = InstantaneousAction('leave', l=loc)
        l = leave.parameter('l')
        leave.add_precondition(at(l))
        leave.add_precondition(goal(l))
        leave.add_effect(at(l), False)
        leave.add_effect(free(l), True)
        leave.add_effect(left(), True)
        leave.add_effect(not_arrived(), False)

        for agent in range(self.num_of_agents):
            carname = "car-" + str(agent)
            car = Agent(carname, self.problem)
            self.problem.add_agent(car)
            car.add_fluent(at, default_initial_value=False)
            car.add_fluent(not_arrived, default_initial_value=True)
            car.add_fluent(start, default_initial_value=False)
            car.add_fluent(left, default_initial_value=False)
            car.add_fluent(goal, default_initial_value=False)
            car.add_action(arrive)
            car.add_action(drive)
            car.add_action(leave)

            slname = self.get_init_loc(agent)
            slobj = unified_planning.model.Object(slname, loc)

            glname = self.get_goal_loc(agent)
            globj = unified_planning.model.Object(glname, loc)

            self.problem.set_initial_value(Dot(car, car.fluent("start")(slobj)), True)
            self.problem.set_initial_value(Dot(car, car.fluent("goal")(globj)), True)

            car.add_public_goal(car.fluent('left'))
        if sl:
            self.add_social_law()
        return self.problem
