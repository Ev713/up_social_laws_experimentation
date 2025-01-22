import copy
import json
import random

import unified_planning
from unified_planning.model import InstantaneousAction, Fluent
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import *

from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


OPERATORS = {
    '=': Equals,
    '>=': GE,
    '>': GT,
    '<=': LE,
    '<': LT
}


class NoSocialLawException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ProblemGenerator():
    def __init__(self):
        self.obj_type = {}
        self.problem = None
        self.instance_data = None
        self.instances_folder = ''

    def generate_problem(self, file_name, sl=False):
        pass

    def add_social_law(self):
        raise NoSocialLawException

    def load_instance_data(self, instance_name):
        json_file_path = self.instances_folder + '/' + instance_name
        with open(json_file_path, 'r') as file:
            self.instance_data = json.load(file)
            return self.instance_data

    def load_objects(self, json_types, obj_types, remember=True):
        for i, obj_type in enumerate(obj_types):
            name = json_types[i]
            self.problem.add_objects(list(map(lambda x: unified_planning.model.Object(x, obj_type),
                                              self.instance_data[name])))
        if remember:
            self.remember_obj_types(json_types, obj_types)

    def load_agents(self):
        for agent_name in self.instance_data['agents']:
            self.problem.add_agent(Agent(agent_name, self.problem))

    def set_init_values(self):
        for key in self.instance_data['init_values']:
            if key == 'global':
                for fluentuple in self.instance_data['init_values'][key]:
                    fluent = self.problem.ma_environment.fluent(fluentuple[0])
                    params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
                    self.problem.set_initial_value(fluent(*params), True)

            else:
                agent = self.problem.agent(key)
                for fluentuple in self.instance_data['init_values'][key]:
                    fluent = fluentuple[0]
                    self.problem.set_initial_value(Dot(agent, agent.fluent(fluent)), True)

    def set_goals(self):
        agent_index = 0
        num_of_agents = len(self.problem.agents)
        for goaltuple in self.instance_data['goals']:
            agent = self.problem.agents[agent_index]
            agent_index = (agent_index + 1) % num_of_agents
            fluent = self.problem.ma_environment.fluent(goaltuple[0])
            params = (unified_planning.model.Object(v, self.obj_type[v]) for v in goaltuple[1])
            agent.add_public_goal(fluent(*params))

    def remember_obj_types(self, json_types, obj_types):
        self.obj_type = {}
        for i, json_type_name in enumerate(json_types):
            for obj_name in self.instance_data[json_type_name]:
                self.obj_type[obj_name] = obj_types[i]


class BlocksworldGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()

    def generate_problem(self, file_name, sl=False):
        self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('blocksworld')

        # Objects
        block = UserType('block')

        self.remember_obj_types(['blocks'], [block])

        # General fluents
        on = Fluent('on', BoolType(), x=block, y=block)
        ontable = Fluent('ontable', BoolType(), x=block)
        clear = Fluent('clear', BoolType(), x=block)
        self.problem.ma_environment.add_fluent(on, default_initial_value=False)
        self.problem.ma_environment.add_fluent(ontable, default_initial_value=False)
        self.problem.ma_environment.add_fluent(clear, default_initial_value=False)

        # Objects
        self.load_objects(['blocks'], [block])

        # Agent specific fluents
        holding = Fluent('holding', BoolType(), x=block)
        handempty = Fluent('handempty', BoolType(), )

        # Actions
        pickup = InstantaneousAction('pick-up', x=block)
        x = pickup.parameter('x')
        pickup.add_precondition(clear(x))
        pickup.add_precondition(ontable(x))
        pickup.add_precondition(handempty())
        pickup.add_effect(ontable(x), False)
        pickup.add_effect(clear(x), False)
        pickup.add_effect(handempty(), False)
        pickup.add_effect(holding(x), True)

        putdown = InstantaneousAction('put-down', x=block)
        x = putdown.parameter('x')
        putdown.add_precondition(holding(x))
        putdown.add_effect(holding(x), False)
        putdown.add_effect(clear(x), True)
        putdown.add_effect(handempty(), True)
        putdown.add_effect(ontable(x), True)

        stack = InstantaneousAction('stack', x=block, y=block)
        x = stack.parameter('x')
        y = stack.parameter('y')
        stack.add_precondition(holding(x))
        stack.add_precondition(clear(y))
        stack.add_effect(holding(x), False)
        stack.add_effect(clear(x), True)
        stack.add_effect(handempty(), True)
        stack.add_effect(on(x, y), True)

        unstack = InstantaneousAction('unstack', x=block, y=block)
        x = unstack.parameter('x')
        y = unstack.parameter('y')
        unstack.add_precondition(on(x, y))
        unstack.add_precondition(clear(x))
        unstack.add_precondition(handempty())
        unstack.add_effect(holding(x), True)
        unstack.add_effect(clear(y), True)
        unstack.add_effect(clear(x), False)
        unstack.add_effect(handempty(), False)
        unstack.add_effect(on(x, y), False)

        # Agents
        for agent_name in self.instance_json['agents']:
            agent = Agent(agent_name, self.problem)
            self.problem.add_agent(agent)
            agent.add_fluent(holding, default_initial_value=False)
            agent.add_fluent(handempty, default_initial_value=False)
            agent.add_action(pickup)
            agent.add_action(putdown)
            agent.add_action(stack)
            agent.add_action(unstack)

        self.set_init_values()

        self.set_goals()
        if sl:
            self.add_social_law()

        return self.problem


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


class DriverLogGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()

    def add_social_law(self):
        driverlog_sl = SocialLaw()
        for agent in self.problem.agents:
            driverlog_sl.add_new_fluent(agent.name, 'trunk_empty', (("t", "truck"),), True)
            driverlog_sl.add_new_fluent(None, f'{agent.name}_can_board', (("t", "truck"),), True)
            driverlog_sl.add_effect(agent.name, 'LOAD-TRUCK', 'trunk_empty', ('truck',), False)
            driverlog_sl.add_effect(agent.name, 'UNLOAD-TRUCK', 'trunk_empty', ('truck',), True)
            driverlog_sl.add_waitfor_annotation(agent.name, 'BOARD-TRUCK', 'at', ('truck', 'loc'))
            driverlog_sl.add_waitfor_annotation(agent.name, 'BOARD-TRUCK', 'empty', ('truck',))

        for agent in self.problem.agents:
            for other_agent in self.problem.agents:
                if other_agent.name == agent.name:
                    continue
                driverlog_sl.add_effect(agent.name, 'BOARD-TRUCK', f'{other_agent.name}_can_board', ('truck',), False)
            driverlog_sl.add_precondition_to_action(agent.name, 'BOARD-TRUCK', f'{agent.name}_can_board', ('truck',))

        for truck in self.problem.objects(UserType('truck', father=UserType('locatable'))):
            driverlog_sl.add_public_goal('empty', (truck.name,))
            driverlog_sl.add_public_goal('empty', (truck.name,))

        # add precondition board driver's truck

        return driverlog_sl.compile(self.problem).problem

    def generate_problem(self, file_name, sl=False):
        self.instance_data = self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('driverlog')

        # Objects
        locatable = UserType('locatable')
        location = UserType('location')
        truck = UserType('truck', father=locatable)
        package = UserType('package', father=locatable)

        self.remember_obj_types(['trucks', 'packages', 'locations'], [truck, package, location])

        # General fluents
        in_ = Fluent('in', BoolType(), obj1=package, obj=truck)
        path = Fluent('path', BoolType(), x=location, y=location)
        empty = Fluent('empty', BoolType(), v=truck)
        at = Fluent('at', BoolType(), obj=locatable, loc=location)
        link = Fluent('link', BoolType(), x=location, y=location)

        self.problem.ma_environment.add_fluent(in_, default_initial_value=False)
        self.problem.ma_environment.add_fluent(path, default_initial_value=False)
        self.problem.ma_environment.add_fluent(empty, default_initial_value=False)
        self.problem.ma_environment.add_fluent(at, default_initial_value=False)
        self.problem.ma_environment.add_fluent(link, default_initial_value=False)

        # Objects
        locations = list(map(lambda l: unified_planning.model.Object(l, location), self.instance_data['locations']))
        self.problem.add_objects(locations)
        trucks = list(map(lambda t: unified_planning.model.Object(t, truck), self.instance_data['trucks']))
        self.problem.add_objects(trucks)
        packages = list(map(lambda p: unified_planning.model.Object(p, package), self.instance_data['packages']))
        self.problem.add_objects(packages)

        # Agent specific fluents
        driving = Fluent('driving', BoolType(), v=truck)
        driver_at = Fluent('driver_at', BoolType(), loc=location)

        # Actions
        load_truck = InstantaneousAction('LOAD-TRUCK', truck=truck, obj=package, loc=location)
        tr = load_truck.parameter('truck')
        o = load_truck.parameter('obj')
        l = load_truck.parameter('loc')
        load_truck.add_precondition(at(tr, l))
        load_truck.add_precondition(at(o, l))
        load_truck.add_precondition(driving(tr))
        load_truck.add_effect(at(o, l), False)
        load_truck.add_effect(in_(o, tr), True)

        unload_truck = InstantaneousAction('UNLOAD-TRUCK', truck=truck, obj=package, loc=location)
        tr = unload_truck.parameter('truck')
        o = unload_truck.parameter('obj')
        l = unload_truck.parameter('loc')
        unload_truck.add_precondition(at(tr, l))
        unload_truck.add_precondition(in_(o, tr))
        unload_truck.add_precondition(driving(tr))
        unload_truck.add_effect(at(o, l), True)
        unload_truck.add_effect(in_(o, tr), False)

        board_truck = InstantaneousAction('BOARD-TRUCK', loc=location, truck=truck)
        l = board_truck.parameter('loc')
        tr = board_truck.parameter('truck')
        board_truck.add_precondition(at(tr, l))
        board_truck.add_precondition(driver_at(l))
        board_truck.add_precondition(empty(tr))
        board_truck.add_effect(driver_at(l), False)
        board_truck.add_effect(driving(tr), True)
        board_truck.add_effect(empty(tr), False)

        disembark_truck = InstantaneousAction('DISEMBARK-TRUCK', loc=location, truck=truck)
        l = disembark_truck.parameter('loc')
        tr = disembark_truck.parameter('truck')
        disembark_truck.add_precondition(at(tr, l))
        disembark_truck.add_precondition(driving(tr))
        disembark_truck.add_effect(driving(tr), False)
        disembark_truck.add_effect(driver_at(l), True)
        disembark_truck.add_effect(empty(tr), True)

        drive_truck = InstantaneousAction('DRIVE-TRUCK', from_=location, to=location, truck=truck)
        from_ = drive_truck.parameter('from_')
        to = drive_truck.parameter('to')
        tr = drive_truck.parameter('truck')
        drive_truck.add_precondition(at(tr, from_))
        drive_truck.add_precondition(driving(tr))
        drive_truck.add_precondition(link(from_, to))
        drive_truck.add_effect(at(tr, to), True)
        drive_truck.add_effect(at(tr, from_), False)

        walk = InstantaneousAction('WALK', from_=location, to=location)
        from_ = walk.parameter('from_')
        to = walk.parameter('to')
        walk.add_precondition(driver_at(from_))
        walk.add_precondition(path(from_, to))
        walk.add_effect(driver_at(to), True)
        walk.add_effect(driver_at(from_), False)

        # Agents
        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            self.problem.add_agent(agent)
            agent.add_fluent(driver_at, default_initial_value=False)
            agent.add_fluent(driving, default_initial_value=False)
            agent.add_action(load_truck)
            agent.add_action(unload_truck)
            agent.add_action(board_truck)
            agent.add_action(disembark_truck)
            agent.add_action(drive_truck)
            agent.add_action(walk)

        self.set_init_values()
        self.set_goals()
        if sl:
            self.add_social_law()
        return self.problem


class ZenoTravelGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()

    def add_social_law(self):
        zenotravel_sl = SocialLaw()
        for agent in self.problem.agents:
            zenotravel_sl.add_new_fluent(agent.name, 'assigned', (('p', 'person'),), False)
        persons_to_aircraft = {}
        for agent in self.problem.agents:
            for goal in agent.public_goals:
                args = [arg.object() for arg in goal.args if arg.is_object_exp()]
                persons_args = [obj for obj in args if obj.type.name == 'person']
                for person in persons_args:
                    persons_to_aircraft[person.name] = agent.name
        for agent in self.problem.agents:
            zenotravel_sl.add_precondition_to_action(agent.name, 'board', 'assigned', ('p',))
        for person in self.problem.objects(UserType('person')):
            if person.name in persons_to_aircraft:
                aircraft_name = persons_to_aircraft[person.name]
                zenotravel_sl.set_initial_value_for_new_fluent(aircraft_name, 'assigned', (person.name,), True)
        return zenotravel_sl.compile(self.problem).problem

    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('Zenotravel')
        self.load_instance_data(file_name)

        # Object types
        city = UserType('city')
        flevel = UserType('flevel')
        person = UserType('person')

        self.remember_obj_types(['citys', 'flevels', 'persons'], [city, flevel, person])

        citys = list(map(lambda c: unified_planning.model.Object(c, city), self.instance_data['citys']))
        self.problem.add_objects(citys)
        flevels = list(map(lambda f: unified_planning.model.Object(f, flevel), self.instance_data['flevels']))
        self.problem.add_objects(flevels)
        persons = list(map(lambda p: unified_planning.model.Object(p, person), self.instance_data['persons']))
        self.problem.add_objects(persons)

        # Public fluents
        person_at = Fluent('person_at', BoolType(), x=person, c=city)
        next = Fluent('next', BoolType(), l1=flevel, l2=flevel)
        self.problem.ma_environment.add_fluent(person_at, default_initial_value=False)
        self.problem.ma_environment.add_fluent(next, default_initial_value=False)

        # Agent fluents
        fuel_level = Fluent('fuel-level', BoolType(), l=flevel)
        carries = Fluent('carries', BoolType(), p=person)
        aircraft_at = Fluent('aircraft_at', BoolType(), c=city)

        # Actions
        board = InstantaneousAction('board', p=person, c=city)
        p = board.parameter('p')
        c = board.parameter('c')
        board.add_precondition(person_at(p, c))
        board.add_precondition(aircraft_at(c))
        board.add_effect(carries(p), True)
        board.add_effect(person_at(p, c), False)

        debark = InstantaneousAction('debark', p=person, c=city)
        p = debark.parameter('p')
        c = debark.parameter('c')
        debark.add_precondition(carries(p))
        debark.add_precondition(aircraft_at(c))
        debark.add_effect(person_at(p, c), True)
        debark.add_effect(carries(p), False)

        fly = InstantaneousAction('fly', c1=city, c2=city, l1=flevel, l2=flevel)
        c1 = fly.parameter('c1')
        c2 = fly.parameter('c2')
        l1 = fly.parameter('l1')
        l2 = fly.parameter('l2')
        fly.add_precondition(aircraft_at(c1))
        fly.add_precondition(fuel_level(l1))
        fly.add_precondition(next(l2, l1))
        fly.add_effect(aircraft_at(c2), True)
        fly.add_effect(fuel_level(l2), True)
        fly.add_effect(aircraft_at(c1), False)
        fly.add_effect(fuel_level(l1), False)

        zoom = InstantaneousAction('zoom', c1=city, c2=city, l1=flevel, l2=flevel, l3=flevel)
        c1 = zoom.parameter('c1')
        c2 = zoom.parameter('c2')
        l1 = zoom.parameter('l1')
        l2 = zoom.parameter('l2')
        l3 = zoom.parameter('l3')
        zoom.add_precondition(aircraft_at(c1))
        zoom.add_precondition(fuel_level(l1))
        zoom.add_precondition(next(l2, l1))
        zoom.add_precondition(next(l3, l2))
        zoom.add_effect(aircraft_at(c2), True)
        zoom.add_effect(fuel_level(l3), True)
        zoom.add_effect(aircraft_at(c1), False)
        zoom.add_effect(fuel_level(l1), True)

        refuel = InstantaneousAction('refuel', c=city, l=flevel, l1=flevel)
        c = refuel.parameter('c')
        l = refuel.parameter('l')
        l1 = refuel.parameter('l1')
        refuel.add_precondition(fuel_level(l))
        refuel.add_precondition(next(l, l1))
        refuel.add_precondition(aircraft_at(c))
        refuel.add_effect(fuel_level(l1), True)
        refuel.add_effect(fuel_level(l), False)

        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            agent.add_fluent(fuel_level, default_initial_value=False)
            agent.add_fluent(carries, default_initial_value=False)
            agent.add_fluent(aircraft_at, default_initial_value=False)
            agent.add_action(board)
            agent.add_action(debark)
            agent.add_action(fly)
            agent.add_action(zoom)
            agent.add_action(refuel)
            self.problem.add_agent(agent)

        self.set_init_values()
        self.set_goals()

        if sl:
            self.add_social_law()

        return self.problem


class NumericProblemGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()
        self.agent_type_name = None

    def set_init_values(self):
        for key in self.instance_data['init_values']:
            for fluentuple in self.instance_data['init_values'][key]:
                value = True
                if fluentuple[0] in OPERATORS:
                    value = float(fluentuple[-1])
                    if value % 1 == 0:
                        value = int(value)
                    fluentuple = fluentuple[1]

                if key == 'global':
                    fluent = self.problem.ma_environment.fluent(fluentuple[0])
                    params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
                    self.problem.set_initial_value(fluent(*params), value)
                else:
                    agent = self.problem.agent(key)
                    fluent = agent.fluent(fluentuple[0])
                    params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
                    agent = self.problem.agent(key)
                    self.problem.set_initial_value(Dot(agent, fluent(*params)), value)

    def set_goals(self):
        assigned_index = 0
        num_of_agents = len(self.problem.agents)
        for agent_name in ['global'] + [agent.name for agent in self.problem.agents]:
            if agent_name not in self.instance_data['goals']:
                continue
            for goaltuple in self.instance_data['goals'][agent_name]:
                if agent_name == 'global':
                    agent = self.problem.agents[assigned_index]
                    assigned_index = (assigned_index + 1) % num_of_agents
                else:
                    agent = self.problem.agent(agent_name)
                if goaltuple[0] in OPERATORS:
                    expr = OPERATORS[goaltuple[0]] \
                        (*[self.create_fluent_expression(goal_expr,
                                                         None if agent_name == 'global' else agent) for goal_expr in
                           goaltuple[1:]])
                    agent.add_public_goal(expr)

                else:
                    agent.add_public_goal(
                        self.create_fluent_expression(goaltuple, None if agent_name == 'global' else agent))

    def create_fluent_expression(self, fluentuple, agent):
        if is_number(str(fluentuple)):
            return float(fluentuple)
        if agent is None:
            fluent = self.problem.ma_environment.fluent(fluentuple[0])
        else:
            fluent = agent.fluent(fluentuple[0])
        params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
        return fluent(*params)


class NumericZenotravelGenerator(NumericProblemGenerator):
    def __init__(self):
        super().__init__()

    def add_social_law(self):
        zenotravel_sl = SocialLaw()
        for agent in self.problem.agents:
            zenotravel_sl.add_new_fluent(agent.name, 'assigned', (('p', 'person'),), False)
        persons_to_aircraft = {}
        for agent in self.problem.agents:
            for goal in agent.public_goals:
                args = [arg.object() for arg in goal.args if arg.is_object_exp()]
                persons_args = [obj for obj in args if obj.type.name == 'person']
                for person in persons_args:
                    persons_to_aircraft[person.name] = agent.name
        for agent in self.problem.agents:
            zenotravel_sl.add_precondition_to_action(agent.name, 'board', 'assigned', ('p',))
        for person in self.problem.objects(UserType('person')):
            if person.name in persons_to_aircraft:
                aircraft_name = persons_to_aircraft[person.name]
                zenotravel_sl.set_initial_value_for_new_fluent(aircraft_name, 'assigned', (person.name,), True)
        zenotravel_sl.skip_checks = True
        self.problem = zenotravel_sl.compile(self.problem).problem
        return self.problem

    def generate_problem(self, file_name, sl=False):
        self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('zenotravel_' + file_name.replace('.json', ''))
        self.agent_type_name = 'plane'

        # Object types
        city = UserType('city')
        person = UserType('person')

        self.remember_obj_types(['city', 'person'], [city, person])
        self.load_objects(['city', 'person'], [city, person])

        # Public fluents
        person_loc = Fluent('person-loc', BoolType(), x=person, c=city)
        self.problem.ma_environment.add_fluent(person_loc, default_initial_value=False)
        distance = Fluent('distance', RealType(), c1=city, c2=city)
        self.problem.ma_environment.add_fluent(distance, default_initial_value=False)

        # Agent fluents
        carries = Fluent('carries', BoolType(), p=person)
        fuel = Fluent('fuel', RealType(), )
        slow_burn = Fluent('slow-burn', RealType(), )
        fast_burn = Fluent('fast-burn', RealType(), )
        capacity = Fluent('capacity', RealType(), )
        onboard = Fluent('onboard', RealType(), )
        zoom_limit = Fluent('zoom-limit', RealType(), )
        aircraft_loc = Fluent('aircraft-loc', BoolType(), c=city)

        # Actions
        board = InstantaneousAction('board', p=person, c=city)
        p = board.parameter('p')
        c = board.parameter('c')
        board.add_precondition(person_loc(p, c))
        board.add_precondition(aircraft_loc(c))
        board.add_effect(onboard, Plus(onboard + 1))
        board.add_effect(carries(p), True)
        board.add_effect(person_loc(p, c), False)

        debark = InstantaneousAction('debark', p=person, c=city)
        p = debark.parameter('p')
        c = debark.parameter('c')
        debark.add_precondition(carries(p))
        debark.add_precondition(aircraft_loc(c))
        debark.add_effect(onboard, Minus(onboard, 1))
        debark.add_effect(person_loc(p, c), True)
        debark.add_effect(carries(p), False)

        fly_slow = InstantaneousAction('fly-slow', c1=city, c2=city)
        c1 = fly_slow.parameter('c1')
        c2 = fly_slow.parameter('c2')
        fly_slow.add_precondition(aircraft_loc(c1))
        fly_slow.add_precondition(GE(fuel, Times(distance(c1, c2), slow_burn)))
        fly_slow.add_precondition(GT(distance(c1, c2), 0))

        fly_slow.add_effect(aircraft_loc(c2), True)
        fly_slow.add_effect(aircraft_loc(c1), False)
        fly_slow.add_effect(fuel, Minus(fuel, Times(distance(c1, c2), slow_burn)))

        fly_fast = InstantaneousAction('fly-fast', c1=city, c2=city, )
        c1 = fly_fast.parameter('c1')
        c2 = fly_fast.parameter('c2')
        fly_fast.add_precondition(aircraft_loc(c1))
        fly_fast.add_precondition(GT(distance(c1, c2), 0))
        fly_fast.add_precondition(GE(fuel, Times(distance(c1, c2), fast_burn)))
        fly_fast.add_precondition(GE(zoom_limit, onboard))
        fly_fast.add_effect(aircraft_loc(c2), True)
        fly_fast.add_effect(aircraft_loc(c1), False)
        fly_fast.add_effect(fuel, Minus(fuel, Times(distance(c1, c2), fast_burn)))

        refuel = InstantaneousAction('refuel', )
        refuel.add_precondition(GT(capacity, fuel))
        refuel.add_effect(fuel, capacity)

        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            agent.add_fluent(fuel, default_initial_value=0)
            agent.add_fluent(carries, default_initial_value=False)
            agent.add_fluent(aircraft_loc, default_initial_value=False)
            agent.add_fluent(capacity, default_initial_value=0)
            agent.add_fluent(fast_burn, default_initial_value=0)
            agent.add_fluent(slow_burn, default_initial_value=0)
            agent.add_fluent(onboard, default_initial_value=0)
            agent.add_fluent(zoom_limit, default_initial_value=0)

            agent.add_action(board)
            agent.add_action(debark)
            agent.add_action(fly_slow)
            agent.add_action(fly_fast)
            agent.add_action(refuel)
            self.problem.add_agent(agent)

        self.set_init_values()
        self.set_goals()

        if sl:
            self.add_social_law()
        return self.problem


class NumericGridGenerator(NumericProblemGenerator):

    def __init__(self):
        super().__init__()
        self.fluent = {
            'agent_x': {},
            'agent_y': {},
            'goal_x': {},
            'goal_y': {},
            'init_x': {},
            'init_y': {},
            'on_map': {},
            'left': {}
        }

    def set_init_values(self):
        for a in self.instance_data['agents']:
            agent_data = self.instance_data[a]
            for f in agent_data:
                fluent = self.fluent[f][a]
                self.problem.set_initial_value(fluent(), agent_data[f])

    def add_social_law(self):
        # Clockwise movement on edges and alternating between up and right everywhere else.
        up_columns = []
        down_columns = []
        l = 0
        r = self.instance_data['max_x']
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
        direction_law.skip_checks = True
        min_x = self.instance_data['min_x']
        max_x = self.instance_data['max_x']
        min_y = self.instance_data['min_y']
        max_y = self.instance_data['max_y']
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if y < max_y:
                    for a in self.problem.agents:
                        a.action('move_right').add_precondition(Not(And(Equals(self.fluent['agent_x'][a.name], x),
                                                                        Equals(self.fluent['agent_y'][a.name], y))))
                if y > min_y:
                    for a in self.problem.agents:
                        a.action('move_left').add_precondition(Not(And(Equals(self.fluent['agent_x'][a.name], x),
                                                                       Equals(self.fluent['agent_y'][a.name], y))))
                if x not in up_columns:
                    for a in self.problem.agents:
                        a.action('move_up').add_precondition(Not(And(Equals(self.fluent['agent_x'][a.name], x),
                                                                     Equals(self.fluent['agent_y'][a.name], y))))
                if x not in down_columns:
                    for a in self.problem.agents:
                        a.action('move_down').add_precondition(Not(And(Equals(self.fluent['agent_x'][a.name], x),
                                                                       Equals(self.fluent['agent_y'][a.name], y))))
        self.problem = direction_law.compile(self.problem).problem
        return self.problem

    def add_is_free_precon(self, action, agent, x, y, waitfor=False):
        # skip = input('Skip?:' )
        # if skip in ['y', 'yes', 'Y',]:
        #    return
        other_agents = [a for a in self.problem.agents if a.name != agent.name]
        for other_agent in other_agents:
            other_x = self.fluent['agent_x'][other_agent.name]()
            other_y = self.fluent['agent_y'][other_agent.name]()
            other_on_map = self.fluent['on_map'][other_agent.name]()
            precon = Not(And(Equals(x, other_x), Equals(y, other_y), other_on_map))
            action.add_precondition(precon)
            if waitfor:
                self.problem.waitfor.annotate_as_waitfor(agent.name, action.name, precon)

    def generate_problem(self, file_name, sl=False):
        self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('grid_' + file_name.replace('.json', ''))

        min_x = self.instance_data['min_x']
        max_x = self.instance_data['max_x']
        min_y = self.instance_data['min_y']
        max_y = self.instance_data['max_y']
        # self.problem.ma_environment.add_fluent(is_free, default_initial_value=True)

        self.load_agents()

        for agent in self.problem.agents:
            # Agent Fluents
            self.fluent['agent_x'][agent.name] = Fluent(f'{agent.name}_x', IntType(), )
            self.fluent['agent_y'][agent.name] = Fluent(f'{agent.name}_y', IntType(), )
            self.fluent['goal_x'][agent.name] = Fluent(f'{agent.name}_goal_x', IntType(), )
            self.fluent['goal_y'][agent.name] = Fluent(f'{agent.name}_goal_y', IntType(), )
            self.fluent['init_x'][agent.name] = Fluent(f'{agent.name}_init_x', IntType(), )
            self.fluent['init_y'][agent.name] = Fluent(f'{agent.name}_init_y', IntType(), )
            self.fluent['on_map'][agent.name] = Fluent(f'{agent.name}_on_map', BoolType(), )
            self.fluent['left'][agent.name] = Fluent(f'{agent.name}_left', BoolType(), )

            self.problem.ma_environment.add_fluent(self.fluent['agent_x'][agent.name], default_initial_value=0)
            self.problem.ma_environment.add_fluent(self.fluent['agent_y'][agent.name], default_initial_value=0)
            self.problem.ma_environment.add_fluent(self.fluent['goal_x'][agent.name], default_initial_value=0)
            self.problem.ma_environment.add_fluent(self.fluent['goal_y'][agent.name], default_initial_value=0)
            self.problem.ma_environment.add_fluent(self.fluent['init_x'][agent.name], default_initial_value=0)
            self.problem.ma_environment.add_fluent(self.fluent['init_y'][agent.name], default_initial_value=0)
            self.problem.ma_environment.add_fluent(self.fluent['on_map'][agent.name], default_initial_value=False)
            self.problem.ma_environment.add_fluent(self.fluent['left'][agent.name], default_initial_value=False)

            # Actions
        for agent in self.problem.agents:
            leave = InstantaneousAction('leave', )

            leave.add_precondition(Equals(self.fluent['agent_x'][agent.name](), self.fluent['goal_x'][agent.name]()))
            leave.add_precondition(Equals(self.fluent['agent_y'][agent.name](), self.fluent['goal_y'][agent.name]()))

            leave.add_effect(self.fluent['left'][agent.name](), True)
            leave.add_effect(self.fluent['on_map'][agent.name](), False)

            appear = InstantaneousAction('appear')
            appear.add_precondition(Not(self.fluent['on_map'][agent.name]()))
            appear.add_precondition(Not(self.fluent['left'][agent.name]()))

            appear.add_effect(self.fluent['agent_x'][agent.name](), self.fluent['init_x'][agent.name]())
            appear.add_effect(self.fluent['agent_y'][agent.name](), self.fluent['init_y'][agent.name]())
            appear.add_effect(self.fluent['on_map'][agent.name](), True)
            agent.add_action(appear)
            self.add_is_free_precon(agent.action('appear'), agent, self.fluent['init_x'][agent.name](),
                                    self.fluent['init_y'][agent.name](), sl)

            x_from_range = {
                'up': (min_x, max_x),
                'down': (min_x, max_x),
                'left': (min_x + 1, max_x),
                'right': (min_x, max_x - 1)
            }

            y_from_range = {
                'up': (min_y, max_y - 1),
                'down': (min_y + 1, max_y),
                'left': (min_y, max_y),
                'right': (min_y, max_y),
            }

            moves = {}
            for d in ['up', 'down', 'left', 'right']:
                move = InstantaneousAction(f'move_{d}')
                move.add_precondition(self.fluent['on_map'][agent.name]())
                move.add_precondition(GE(self.fluent['agent_x'][agent.name](), x_from_range[d][0]))
                move.add_precondition(LE(self.fluent['agent_x'][agent.name](), x_from_range[d][1]))
                move.add_precondition(GE(self.fluent['agent_y'][agent.name](), y_from_range[d][0]))
                move.add_precondition(LE(self.fluent['agent_y'][agent.name](), y_from_range[d][1]))

                effect = {
                    'right': [self.fluent['agent_x'][agent.name](), Plus(self.fluent['agent_x'][agent.name](), 1)],
                    'left': [self.fluent['agent_x'][agent.name](), Minus(self.fluent['agent_x'][agent.name](), 1)],
                    'up': [self.fluent['agent_y'][agent.name](), Plus(self.fluent['agent_y'][agent.name](), 1)],
                    'down': [self.fluent['agent_y'][agent.name](), Minus(self.fluent['agent_y'][agent.name](), 1)]
                }[d]
                move.add_effect(*effect)
                moves[d] = move

            for d in ['up', 'down', 'left', 'right']:
                agent.add_action(moves[d])
                if d == 'up':
                    self.add_is_free_precon(moves[d], agent, self.fluent['agent_x'][agent.name](),
                                            Plus(self.fluent['agent_y'][agent.name](), 1), sl)
                if d == 'down':
                    self.add_is_free_precon(moves[d], agent, self.fluent['agent_x'][agent.name](),
                                            Minus(self.fluent['agent_y'][agent.name](), 1), sl)
                if d == 'left':
                    self.add_is_free_precon(moves[d], agent,
                                            Minus(self.fluent['agent_x'][agent.name](), 1),
                                            self.fluent['agent_y'][agent.name](), sl)
                if d == 'right':
                    self.add_is_free_precon(moves[d], agent,
                                            Plus(self.fluent['agent_x'][agent.name](), 1),
                                            self.fluent['agent_y'][agent.name](), sl)
                #

            agent.add_action(leave)

            agent.add_public_goal(self.fluent['left'][agent.name]())

        self.set_init_values()
        if sl:
            self.add_social_law()
        return self.problem


class ExpeditionGenerator(NumericProblemGenerator):

    def add_social_law(self):
        sl = SocialLaw()
        sl.skip_checks = True
        packs = self.count_packs_needed()
        starting_loc = {}
        for a in self.problem.agents:
            print(a.name)
            starting_loc[a.name] = None
            for w in self.instance_data['waypoint']:
                init_values = self.instance_data['init_values'][a.name]
                for init_val in init_values:
                    if init_val[0] == 'at' and len(init_val) > 0 and w in init_val[1]:
                        starting_loc[a.name] = w
                        break
                    if starting_loc[a.name] is not None:
                        break
            if starting_loc[a.name] is None:
                raise Exception(f'Agent {a.name} doesn\'t have a starting location')
            print(f'starting loc: {starting_loc[a.name]}')
        for a in self.problem.agents:
            packs = None
            for init_val in self.instance_data['init_values']:
                if init_val[0] == '=' and init_val[1][1][0] == starting_loc[a.name]:
                    packs = int(int(init_val[2])/len([x for x in starting_loc if starting_loc[x] == starting_loc[a.name]]))
                    print(f'{a.name} packs: {packs}')
                    break
            if packs == None:
                raise Exception(f'Can\'t find Agent {a.name}\' packs!')
            sl.add_new_fluent(a.name, 'personal_packs', (('w', 'waypoint'), ), 0)
            sl.add_precondition_to_action(a.name, 'retrieve_supplies', 'personal_packs', ('w',), '>=', 1)
            sl.add_effect(a.name, 'retrieve_supplies', 'personal_packs', ('w',), 1, '-')
            sl.add_effect(a.name, 'store_supplies', 'personal_packs', ('w',), 1, '+')
            sl.set_initial_value_for_new_fluent(a.name, 'personal_packs', (starting_loc[a.name],), packs)
        self.problem = sl.compile(self.problem).problem
        return self.problem

    def count_packs_needed(self):
        num_of_waypoints = len([w for w in self.problem.objects(self.obj_type['wa0'])])
        s = 0
        for i in range(0, num_of_waypoints):
            if i <= 4:
                s = i
                continue
            if s % 2 == 0:
                s = 2 * s - 1
            else:
                s = 2 * s
        return s

    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('expedition_' + file_name.replace('.json', ''))
        self.load_instance_data(file_name)

        waypoint = UserType('waypoint')
        at = Fluent('at', BoolType(), w=waypoint)
        is_next = Fluent('is_next', BoolType(), x=waypoint, y=waypoint)
        sled_supplies = Fluent('sled_supplies', IntType())
        sled_capacity = Fluent('sled_capacity', IntType())
        waypoint_supplies = Fluent('waypoint_supplies', IntType(), w=waypoint)

        self.load_objects(['waypoint'], [waypoint])

        self.problem.ma_environment.add_fluent(is_next, default_initial_value=False)
        self.problem.ma_environment.add_fluent(waypoint_supplies, default_initial_value=0)

        move = {}
        for dir in ['forwards', 'backwards']:
            move[dir] = InstantaneousAction(f'move_{dir}', w1=waypoint, w2=waypoint)
            w1 = move[dir].parameter('w1')
            w2 = move[dir].parameter('w2')
            move[dir].add_precondition(at(w1))
            if dir == 'forwards':
                move[dir].add_precondition(is_next(w1, w2))
            else:
                move[dir].add_precondition(is_next(w2, w1))
            move[dir].add_precondition(GE(sled_supplies, 1))
            move[dir].add_effect(at(w1), False)
            move[dir].add_effect(at(w2), True)
            move[dir].add_effect(sled_supplies, Minus(sled_supplies, 1))

        store_supplies = InstantaneousAction('store_supplies', w=waypoint)
        w = store_supplies.parameter('w')
        store_supplies.add_precondition(at(w))
        store_supplies.add_precondition(GE(sled_supplies, 1))
        store_supplies.add_effect(sled_supplies, Minus(sled_supplies, 1))
        store_supplies.add_effect(waypoint_supplies(w), Plus(waypoint_supplies(w), 1))

        retrieve_supplies = InstantaneousAction('retrieve_supplies', w=waypoint)
        w = retrieve_supplies.parameter('w')
        retrieve_supplies.add_precondition(at(w))
        retrieve_supplies.add_precondition(GE(waypoint_supplies(w), 1))
        retrieve_supplies.add_precondition(GT(sled_capacity, sled_supplies))
        retrieve_supplies.add_effect(sled_supplies, Plus(sled_supplies, 1))
        retrieve_supplies.add_effect(waypoint_supplies(w), Minus(waypoint_supplies(w), 1))

        self.load_agents()
        for a in self.problem.agents:
            a.add_action(move['forwards'])
            a.add_action(move['backwards'])
            a.add_action(store_supplies)
            a.add_action(retrieve_supplies)
            a.add_fluent(sled_supplies, default_initial_value=0)
            a.add_fluent(sled_capacity, default_initial_value=0)
            a.add_fluent(at, default_initial_value=False)

        self.set_init_values()
        self.set_goals()
        if sl:
            self.add_social_law()
        return self.problem


class MarketTraderGenerator(NumericProblemGenerator):
    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('market_trader_' + file_name.replace('.json', ''))
        self.load_instance_data(file_name)
        market = UserType('market')
        goods = UserType('goods')
        self.load_objects(['market', 'goods'], [market, goods])
        on_sale = Fluent('on-sale', IntType(), g=goods, m=market)
        drive_cost = Fluent('drive-cost', RealType(), m1=market, m2=market)
        price = Fluent('price', RealType(), g=goods, m=market)
        sellprice = Fluent('sellprice', RealType(), g=goods, m=market)
        self.problem.ma_environment.add_fluent(on_sale, default_initial_value=0)
        self.problem.ma_environment.add_fluent(drive_cost, default_initial_value=0)
        self.problem.ma_environment.add_fluent(price, default_initial_value=0)
        self.problem.ma_environment.add_fluent(sellprice, default_initial_value=0)

        bought = Fluent('bought', IntType(), g=goods)
        cash = Fluent('cash', RealType())
        capacity = Fluent('capacity', IntType())
        at = Fluent('at', BoolType(), m=market)
        can_drive = Fluent('can-drive', BoolType(), m1=market, m2=market)

        travel = InstantaneousAction('travel', m1=market, m2=market)
        m1 = travel.parameter('m1')
        m2 = travel.parameter('m2')
        travel.add_precondition(can_drive(m1, m2))
        travel.add_precondition(GE(cash, drive_cost(m1, m2)))
        travel.add_precondition(at(m1))
        travel.add_effect(cash, Minus(cash, drive_cost(m1, m2)))
        travel.add_effect(at(m1), False)
        travel.add_effect(at(m2), True)

        buy = InstantaneousAction('buy', g=goods, m=market)
        g = buy.parameter('g')
        m = buy.parameter('m')
        buy.add_precondition(at(m))
        buy.add_precondition(LE(price(g, m), cash))
        buy.add_precondition(GE(capacity, 1))
        buy.add_precondition(GT(on_sale(g, m), 0))
        buy.add_effect(capacity, Minus(capacity, 1))
        buy.add_effect(on_sale(g, m), Minus(on_sale(g, m), 1))
        buy.add_effect(bought(g), Plus(bought(g), 1))
        buy.add_effect(cash, Minus(cash, price(g, m)))

        upgrade = InstantaneousAction('upgrade', )
        upgrade.add_precondition(GE(cash, 5))
        upgrade.add_effect(cash, Minus(cash, 50))
        upgrade.add_effect(capacity, Plus(capacity, 20))

        sell = InstantaneousAction('sell', g=goods, m=market)
        sell.add_precondition(at(m))
        sell.add_precondition(GE(bought(g), 1))
        sell.add_effect(capacity, Plus(capacity, 1))
        sell.add_effect(bought(g), Minus(bought(g), 1))
        sell.add_effect(on_sale(g, m), Plus(on_sale(g, m), 1))
        sell.add_effect(cash, Plus(cash, sellprice(g, m)))
        self.load_agents()
        for a in self.problem.agents:
            a.add_fluent(at, default_initial_value=False)
            a.add_fluent(can_drive, default_initial_value=False)
            a.add_fluent(bought, default_initial_value=0)
            a.add_fluent(cash, default_initial_value=0)
            a.add_fluent(capacity, default_initial_value=0)
            a.add_action(travel)
            a.add_action(buy)
            a.add_action(upgrade)
            a.add_action(sell)

        self.set_init_values()
        self.set_goals()
        if sl:
            self.add_social_law()
        return self.problem

    def set_goals(self):
        assigned_index = 0
        num_of_agents = len(self.problem.agents)
        for agent_name in ['global'] + [agent.name for agent in self.problem.agents]:
            if agent_name not in self.instance_data['goals']:
                continue
            for goaltuple in self.instance_data['goals'][agent_name]:
                if agent_name == 'global':
                    agent = self.problem.agents[assigned_index]
                    assigned_index = (assigned_index + 1) % num_of_agents
                else:
                    agent = self.problem.agent(agent_name)
                if goaltuple[0] in OPERATORS:
                    expr = OPERATORS[goaltuple[0]] \
                        (*[self.create_fluent_expression(goal_expr,
                                                         None if agent_name == 'global' else agent) for goal_expr in
                           goaltuple[1:]])
                    agent.add_public_goal(expr)

                else:
                    agent.add_public_goal(
                        self.create_fluent_expression(goaltuple, None if agent_name == 'global' else agent))


if __name__ == '__main__':
    pg = ExpeditionGenerator()
    pg.instances_folder = './numeric_problems/expedition/json'
    prob = pg.generate_problem('pfile10.json', sl=True)
    print(prob)
