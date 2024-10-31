import csv
import json
from datetime import date

import pandas as pandas
import unified_planning
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
import random
from unified_planning.io import PDDLWriter, MAPDDLWriter

up.shortcuts.get_environment().credits_stream = None

# from test.test_social_law import Example
from up_social_laws.ma_centralizer import MultiAgentProblemCentralizer
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.robustness_verification import WaitingActionRobustnessVerifier
from up_social_laws.single_agent_projection import SingleAgentProjection
from up_social_laws.social_law import SocialLaw

from up_social_laws.robustness_checker import SocialLawRobustnessChecker
from unified_planning.io import PDDLReader
import resource
import time
import signal
import os
from multiprocessing import Process, Queue

planner = OneshotPlanner()

INIT_LOCS = [{0: '(1, 1)', 1: '(1, 1)'}, {0: '(2, 0)', 1: '(0, 2)', 2: '(1, 0)'},
             {0: '(0, 2)', 1: '(1, 1)', 2: '(2, 3)', 3: '(1, 2)'},
             {0: '(0, 2)', 1: '(1, 3)', 2: '(3, 0)', 3: '(2, 1)', 4: '(0, 3)'},
             {0: '(3, 2)', 1: '(3, 1)', 2: '(0, 3)', 3: '(3, 2)', 4: '(0, 0)', 5: '(0, 3)'},
             {0: '(1, 1)', 1: '(0, 0)', 2: '(2, 2)', 3: '(1, 3)', 4: '(3, 4)', 5: '(3, 0)'},
             {0: '(3, 1)', 1: '(2, 4)', 2: '(4, 1)', 3: '(2, 0)', 4: '(3, 0)', 5: '(1, 4)', 6: '(3, 0)'},
             {0: '(2, 5)', 1: '(1, 0)', 2: '(5, 4)', 3: '(5, 0)', 4: '(0, 4)', 5: '(1, 5)', 6: '(2, 1)', 7: '(0, 4)'},
             {0: '(5, 3)', 1: '(1, 2)', 2: '(1, 3)', 3: '(2, 2)', 4: '(1, 6)', 5: '(1, 1)', 6: '(4, 3)', 7: '(4, 1)'},
             {0: '(3, 0)', 1: '(4, 2)', 2: '(1, 1)', 3: '(6, 6)', 4: '(5, 5)', 5: '(4, 0)', 6: '(5, 6)'},
             {0: '(2, 7)', 1: '(5, 5)', 2: '(6, 0)', 3: '(1, 3)', 4: '(0, 7)', 5: '(0, 3)', 6: '(4, 5)', 7: '(1, 2)',
              8: '(0, 1)'}, {0: '(6, 1)', 1: '(3, 0)'}, {0: '(2, 2)', 1: '(5, 2)', 2: '(2, 0)'},
             {0: '(4, 1)', 1: '(2, 0)', 2: '(0, 1)', 3: '(2, 3)', 4: '(0, 1)', 5: '(3, 2)'},
             {0: '(0, 1)', 1: '(6, 1)', 2: '(0, 0)', 3: '(0, 4)', 4: '(5, 2)', 5: '(2, 2)', 6: '(5, 1)'},
             {0: '(0, 4)', 1: '(7, 2)', 2: '(3, 2)', 3: '(4, 2)', 4: '(0, 3)', 5: '(5, 1)', 6: '(7, 3)', 7: '(6, 4)'},
             {0: '(6, 6)', 1: '(4, 1)', 2: '(7, 1)', 3: '(5, 6)', 4: '(0, 4)', 5: '(1, 0)', 6: '(1, 1)', 7: '(2, 4)',
              8: '(1, 1)', 9: '(5, 0)'},
             {0: '(3, 3)', 1: '(3, 4)', 2: '(5, 3)', 3: '(4, 2)', 4: '(2, 7)', 5: '(1, 7)', 6: '(5, 4)', 7: '(1, 1)'},
             {0: '(4, 6)', 1: '(2, 4)', 2: '(2, 5)', 3: '(2, 6)', 4: '(3, 1)', 5: '(4, 6)', 6: '(3, 7)', 7: '(4, 1)'},
             {0: '(1, 4)', 1: '(0, 1)', 2: '(1, 1)'}]
GOAL_LOCS = [{0: '(1, 2)', 1: '(0, 1)'}, {0: '(0, 2)', 1: '(2, 0)', 2: '(2, 2)'},
             {0: '(1, 3)', 1: '(1, 3)', 2: '(2, 0)', 3: '(0, 0)'},
             {0: '(2, 3)', 1: '(2, 3)', 2: '(3, 0)', 3: '(2, 1)', 4: '(3, 3)'},
             {0: '(0, 4)', 1: '(3, 4)', 2: '(3, 2)', 3: '(2, 3)', 4: '(0, 1)', 5: '(1, 3)'},
             {0: '(1, 1)', 1: '(3, 3)', 2: '(2, 4)', 3: '(4, 1)', 4: '(4, 2)', 5: '(0, 2)'},
             {0: '(4, 2)', 1: '(3, 1)', 2: '(3, 1)', 3: '(3, 1)', 4: '(1, 1)', 5: '(0, 0)', 6: '(2, 1)'},
             {0: '(1, 3)', 1: '(3, 1)', 2: '(5, 3)', 3: '(0, 1)', 4: '(4, 5)', 5: '(5, 0)', 6: '(1, 1)', 7: '(4, 2)'},
             {0: '(3, 3)', 1: '(5, 2)', 2: '(3, 4)', 3: '(4, 1)', 4: '(3, 3)', 5: '(3, 6)', 6: '(1, 0)', 7: '(5, 4)'},
             {0: '(4, 3)', 1: '(5, 2)', 2: '(0, 2)', 3: '(1, 0)', 4: '(1, 5)', 5: '(6, 4)', 6: '(2, 1)'},
             {0: '(3, 1)', 1: '(4, 4)', 2: '(3, 3)', 3: '(1, 7)', 4: '(3, 2)', 5: '(1, 3)', 6: '(6, 2)', 7: '(6, 4)',
              8: '(1, 7)'}, {0: '(1, 1)', 1: '(5, 0)'}, {0: '(4, 2)', 1: '(0, 0)', 2: '(5, 0)'},
             {0: '(7, 1)', 1: '(5, 3)', 2: '(4, 1)', 3: '(2, 1)', 4: '(6, 2)', 5: '(0, 1)'},
             {0: '(6, 0)', 1: '(4, 4)', 2: '(7, 2)', 3: '(0, 3)', 4: '(3, 1)', 5: '(5, 0)', 6: '(2, 2)'},
             {0: '(5, 5)', 1: '(5, 5)', 2: '(7, 3)', 3: '(1, 3)', 4: '(5, 2)', 5: '(6, 1)', 6: '(6, 4)', 7: '(0, 0)'},
             {0: '(5, 2)', 1: '(0, 2)', 2: '(7, 6)', 3: '(1, 5)', 4: '(6, 5)', 5: '(0, 3)', 6: '(4, 1)', 7: '(7, 5)',
              8: '(7, 1)', 9: '(0, 6)'},
             {0: '(0, 2)', 1: '(0, 2)', 2: '(4, 7)', 3: '(5, 6)', 4: '(4, 0)', 5: '(5, 5)', 6: '(0, 0)', 7: '(5, 5)'},
             {0: '(4, 5)', 1: '(3, 4)', 2: '(2, 6)', 3: '(2, 3)', 4: '(2, 3)', 5: '(2, 1)', 6: '(2, 5)', 7: '(0, 5)'},
             {0: '(1, 4)', 1: '(2, 3)', 2: '(2, 3)'}]


class GridManager:
    def __init__(self, width, height, agents):
        self.width = width
        self.height = height
        self.num_of_agents = agents
        self.init_locs = {}  # {a: (0, 0) for a in range(self.num_of_agents)}
        self.goal_locs = {}  # {a: (self.width - 1, self.height - 1) for a in range(self.num_of_agents)}
        self.intersections = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.compass = {(0, 1): 'north', (1, 0): 'east', (0, -1): 'south', (-1, 0): 'west'}
        self.directions = {str((l1, l2)): self.get_dir(l1, l2) for l1 in self.intersections for l2 in self.intersections
                           if (l2[0] - l1[0], l2[1] - l1[1]) in self.compass}

        self.planner = planner
        self.slrc = SocialLawRobustnessChecker(
            planner=planner,
            robustness_verifier_name="WaitingActionRobustnessVerifier")

    def get_dir(self, l1, l2):
        diff = (l2[0] - l1[0], l2[1] - l1[1])
        return self.compass[diff]

    def get_grid_problem(self):
        problem = MultiAgentProblemWithWaitfor()
        loc = UserType("loc")

        direction = UserType("direction")
        connected = Fluent('connected', BoolType(), l1=loc, l2=loc, d=direction)
        free = Fluent('free', BoolType(), l=loc)
        problem.ma_environment.add_fluent(connected, default_initial_value=False)
        problem.ma_environment.add_fluent(free, default_initial_value=True)
        problem.add_objects(list(map(lambda d: unified_planning.model.Object(d, direction),
                                     list(self.compass.values()))))
        problem.add_objects(
            list(map(lambda l: unified_planning.model.Object(l, loc), [str(x) for x in self.intersections])))

        for l1 in self.intersections:
            for l2 in self.intersections:
                if (l2[0] - l1[0], l2[1] - l1[1]) not in self.compass:
                    continue
                problem.set_initial_value(
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
            car = Agent(carname, problem)
            problem.add_agent(car)
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

            problem.set_initial_value(Dot(car, car.fluent("start")(slobj)), True)
            problem.set_initial_value(Dot(car, car.fluent("goal")(globj)), True)

            car.add_public_goal(car.fluent('left'))
        return problem

    def get_init_loc(self, agent):
        if agent in self.init_locs and self.init_locs[agent] in self.intersections:
            return str(self.init_locs[agent])
        else:
            self.init_locs[agent] = str(random.choice(self.intersections))
            return self.init_locs[agent]

    def get_goal_loc(self, agent):
        if agent in self.goal_locs and self.goal_locs[agent] in self.intersections:
            return str(self.goal_locs[agent])
        else:
            self.goal_locs[agent] = str(random.choice(self.intersections))
            return self.goal_locs[agent]

    def add_direction_law(self, problem):
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

        return direction_law.compile(problem).problem

    def add_order_law(self, problem):
        order_law = SocialLaw()
        for agent in range(self.num_of_agents):
            order_law.add_new_fluent(None, f'moves_{agent}', (), True)
        for agent in range(self.num_of_agents):
            carname = "car-" + str(agent)
            order_law.add_precondition_to_action(carname, 'arrive', f'moves_{agent}', ())
            order_law.add_precondition_to_action(carname, 'drive', f'moves_{agent}', ())
            order_law.add_precondition_to_action(carname, 'leave', f'moves_{agent}', ())
            order_law.add_waitfor_annotation(carname, 'arrive', f'moves_{agent}', ())
            order_law.add_waitfor_annotation(carname, 'drive', f'moves_{agent}', ())
            order_law.add_waitfor_annotation(carname, 'leave', f'moves_{agent}', ())

        for agent in range(self.num_of_agents):
            for other_agent in range(self.num_of_agents):
                if agent == other_agent:
                    continue
                order_law.add_effect("car-" + str(agent), 'leave', f'moves_{other_agent}', (), True)
        for agent in range(self.num_of_agents):
            for other_agent in range(self.num_of_agents):
                if agent == other_agent:
                    continue
                order_law.add_effect("car-" + str(agent), 'arrive', f'moves_{other_agent}', (), False)
        return order_law.compile(problem).problem


def check_solvable(problem, ma=False):
    if ma:
        problem = ma_to_sa(problem)

    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)
        return result.status


def ma_to_sa(problem):
    return MultiAgentProblemCentralizer().compile(problem).problem


def ma_to_sap(problem, agent_id):
    return SingleAgentProjection(problem.agents[agent_id]).compile(problem).problem


def state_to_dict(state):
    str_state = str(state).replace('{', '').replace('}', '')
    keys = [k for k in
            str_state.replace(' ', '').replace(':', '').replace(',', '').replace('true', '@').replace('false',
                                                                                                      '@').split('@')]
    keys.remove('')
    vals = []
    for word in str_state.replace('true', '@true@').replace('false', '@false@').split('@'):
        if word == 'true':
            vals.append(True)
        if word == 'false':
            vals.append(False)
    state_dict = {}
    for i in range(len(keys)):
        state_dict[keys[i]] = vals[i]
    return state_dict


def state_to_str(state):
    state_dict = state_to_dict(state)
    for key in state_dict:
        print(f'{key}: {state_dict[key]}')


def get_compiled_problem(problem):
    rbv = Compiler(
        name=get_new_slrc()._robustness_verifier_name,
        problem_kind=problem.kind,
        compilation_kind=CompilationKind.MA_SL_ROBUSTNESS_VERIFICATION)

    return rbv.compile(problem).problem


def check_single_agent_solvable(problem):
    for agent in problem.agents:
        sap = SingleAgentProjection(agent)
        result = sap.compile(problem)
        planner = OneshotPlanner(problem_kind=result.problem.kind)
        presult = planner.solve(result.problem)
        if presult.status not in unified_planning.engines.results.POSITIVE_OUTCOMES:
            return False
    return True


def simulate(problem, ma=False, print_state=False, trace_vars=[]):
    with SequentialSimulator(problem) as simulator:
        state = simulator.get_initial_state()
        if print_state:
            print(state_to_str(state))
        if len(trace_vars) > 0:
            for var in trace_vars:
                state_dit = state_to_dict(state)
                print(f'{var}: {state_dit[var]}')
        t = 0
        print(f't = {t}\nCalculating actions...')
        actions = [a for a in simulator.get_applicable_actions(state)]
        if len(actions) == 0:
            print('No legal actions')
            return
        print('Actions: ')
        for a in actions:
            print(a[0].name, a[1])
        while True:
            try:
                while True:
                    choice = int(input('Choice: '))
                    if choice not in range(len(actions)):
                        print('Invalid index. Try again.')
                        continue
                    action = actions[int(choice)]
                    state = simulator.apply(state, action[0], action[1])
                    break
            except:
                print('Applying action failed')
                return
            print(f'Action {action[0].name, action[1]} applied.')

            if simulator.is_goal(state):
                print("Goal reached!")
                return

            t += 1
            actions = [a for a in simulator.get_applicable_actions(state)]
            if len(actions) == 0:
                print('No legal actions')
                return
            print(f't = {t}\nActions:')
            for a in actions:
                print(a[0].name, a[1])


def solve(self, problem, ma=False):
    if ma:
        problem = self.ma_to_sa(problem)
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)
        print(result)


def print_solution(problem):
    mac = MultiAgentProblemCentralizer()
    problem = mac.compile(problem).problem
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)
        print(result)


def print_robustness(problem):
    slrc = SocialLawRobustnessChecker()
    print(slrc.is_robust(problem).status)


def centralise(problem):
    mac = MultiAgentProblemCentralizer()
    return mac.compile(problem).problem


def get_blocksworld(name):
    json_file_path = f'/home/evgeny/SocialLaws/up-social-laws/experimentation/problems/all/jsons/blocksworld/{name}.json'

    # Open the JSON file and load its contents into a dictionary
    with open(json_file_path, 'r') as file:
        instance = json.load(file)

    blocksworld = MultiAgentProblemWithWaitfor('blocksworld')

    # Objects
    block = UserType('block')

    # General fluents
    on = Fluent('on', BoolType(), x=block, y=block)
    ontable = Fluent('ontable', BoolType(), x=block)
    clear = Fluent('clear', BoolType(), x=block)
    blocksworld.ma_environment.add_fluent(on, default_initial_value=False)
    blocksworld.ma_environment.add_fluent(ontable, default_initial_value=False)
    blocksworld.ma_environment.add_fluent(clear, default_initial_value=False)

    # Objects
    locations = list(map(lambda b: unified_planning.model.Object(b, block), instance['blocks']))
    blocksworld.add_objects(locations)

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
    for agent_name in instance['agents']:
        agent = Agent(agent_name, blocksworld)
        blocksworld.add_agent(agent)
        agent.add_fluent(holding, default_initial_value=False)
        agent.add_fluent(handempty, default_initial_value=False)
        agent.add_action(pickup)
        agent.add_action(putdown)
        agent.add_action(stack)
        agent.add_action(unstack)

    for key in instance['init_values']:
        if key == 'global':
            for fluentuple in instance['init_values'][key]:
                fluent = blocksworld.ma_environment.fluent(fluentuple[0])
                params = (unified_planning.model.Object(v, block) for v in fluentuple[1])
                blocksworld.set_initial_value(fluent(*params), True)

        else:
            agent = blocksworld.agent(key)
            for fluentuple in instance['init_values'][key]:
                fluent = fluentuple[0]
                blocksworld.set_initial_value(Dot(agent, agent.fluent(fluent)), True)

    for goaltuple in instance['goals']:
        # for agent in blocksworld.agents:
        fluent = blocksworld.ma_environment.fluent(goaltuple[0])
        params = (unified_planning.model.Object(v, block) for v in goaltuple[1])
        blocksworld.add_goal(fluent(*params))
        # agent.add_public_goal(fluent(*params))

    return blocksworld


def get_driverlog(name):
    json_file_path = f'/home/evgeny/SocialLaws/up-social-laws/experimentation/problems/all/jsons/driverlog/{name}.json'

    # Open the JSON file and load its contents into a dictionary
    with open(json_file_path, 'r') as file:
        instance = json.load(file)

    driverlog = MultiAgentProblemWithWaitfor('blocksworld')

    # Objects
    locatable = UserType('locatable')
    location = UserType('location')
    truck = UserType('truck', father=locatable)
    package = UserType('package', father=locatable)

    obj_type = {}
    for objname in instance['trucks']:
        obj_type[objname] = truck
    for objname in instance['packages']:
        obj_type[objname] = package
    for objname in instance['locations']:
        obj_type[objname] = location

    # General fluents
    in_ = Fluent('in', BoolType(), obj1=package, obj=truck)
    path = Fluent('path', BoolType(), x=location, y=location)
    empty = Fluent('empty', BoolType(), v=truck)
    at = Fluent('at', BoolType(), obj=locatable, loc=location)
    link = Fluent('link', BoolType(), x=location, y=location)

    driverlog.ma_environment.add_fluent(in_, default_initial_value=False)
    driverlog.ma_environment.add_fluent(path, default_initial_value=False)
    driverlog.ma_environment.add_fluent(empty, default_initial_value=False)
    driverlog.ma_environment.add_fluent(at, default_initial_value=False)
    driverlog.ma_environment.add_fluent(link, default_initial_value=False)

    # Objects
    locations = list(map(lambda l: unified_planning.model.Object(l, location), instance['locations']))
    driverlog.add_objects(locations)
    trucks = list(map(lambda t: unified_planning.model.Object(t, truck), instance['trucks']))
    driverlog.add_objects(trucks)
    packages = list(map(lambda p: unified_planning.model.Object(p, package), instance['packages']))
    driverlog.add_objects(packages)

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
    for agent_name in instance['agents']:
        agent = Agent(agent_name, driverlog)
        driverlog.add_agent(agent)
        agent.add_fluent(driver_at, default_initial_value=False)
        agent.add_fluent(driving, default_initial_value=False)
        agent.add_action(load_truck)
        agent.add_action(unload_truck)
        agent.add_action(board_truck)
        agent.add_action(disembark_truck)
        agent.add_action(drive_truck)
        agent.add_action(walk)

    for key in instance['init_values']:
        if key == 'global':
            for fluentuple in instance['init_values'][key]:
                fluent = driverlog.ma_environment.fluent(fluentuple[0])
                params = (unified_planning.model.Object(v, obj_type[v]) for v in fluentuple[1])
                driverlog.set_initial_value(fluent(*params), True)

        else:
            agent = driverlog.agent(key)
            for fluentuple in instance['init_values'][key]:
                fluent = fluentuple[0]
                params = (unified_planning.model.Object(v, obj_type[v]) for v in fluentuple[1])
                driverlog.set_initial_value(Dot(agent, agent.fluent(fluent)(*params)), True)

    agent_index = 0
    num_of_agents = len(driverlog.agents)
    for goaltuple in instance['goals']:
        agent = driverlog.agents[agent_index]
        agent_index = (agent_index + 1) % num_of_agents
        fluent = driverlog.ma_environment.fluent(goaltuple[0])
        params = (unified_planning.model.Object(v, obj_type[v]) for v in goaltuple[1])
        agent.add_public_goal(fluent(*params))
    return driverlog


def driverlog_add_sl(driverlog):
    driverlog_sl = SocialLaw()
    for agent in driverlog.agents:
        driverlog_sl.add_new_fluent(agent.name, 'trunk_empty', (("t", "truck"),), True)
        driverlog_sl.add_new_fluent(None, f'{agent.name}_can_board', (("t", "truck"),), True)
        driverlog_sl.add_effect(agent.name, 'LOAD-TRUCK', 'trunk_empty', ('truck',), False)
        driverlog_sl.add_effect(agent.name, 'UNLOAD-TRUCK', 'trunk_empty', ('truck',), True)
        driverlog_sl.add_waitfor_annotation(agent.name, 'BOARD-TRUCK', 'at', ('truck', 'loc'))
        driverlog_sl.add_waitfor_annotation(agent.name, 'BOARD-TRUCK', 'empty', ('truck',))

    for agent in driverlog.agents:
        for other_agent in driverlog.agents:
            if other_agent.name == agent.name:
                continue
            driverlog_sl.add_effect(agent.name, 'BOARD-TRUCK', f'{other_agent.name}_can_board', ('truck',), False)
        driverlog_sl.add_precondition_to_action(agent.name, 'BOARD-TRUCK', f'{agent.name}_can_board', ('truck',))

    for truck in driverlog.objects(UserType('truck', father=UserType('locatable'))):
        driverlog_sl.add_public_goal('empty', (truck.name,))
        driverlog_sl.add_public_goal('empty', (truck.name,))

    # add precondition board driver's truck

    return driverlog_sl.compile(driverlog).problem


def get_zenotravel(name):
    filepath = open(
        f"/home/evgeny/SocialLaws/up-social-laws/experimentation/problems/all/jsons/zenotravel/{name}.json").read()
    instance = json.loads(filepath)
    zenotravel = MultiAgentProblemWithWaitfor()

    # Object types
    city = UserType('city')
    flevel = UserType('flevel')
    person = UserType('person')

    obj_type = {}
    for objname in instance['citys']:
        obj_type[objname] = city
    for objname in instance['flevels']:
        obj_type[objname] = flevel
    for objname in instance['persons']:
        obj_type[objname] = person

    citys = list(map(lambda c: unified_planning.model.Object(c, city), instance['citys']))
    zenotravel.add_objects(citys)
    flevels = list(map(lambda f: unified_planning.model.Object(f, flevel), instance['flevels']))
    zenotravel.add_objects(flevels)
    persons = list(map(lambda p: unified_planning.model.Object(p, person), instance['persons']))
    zenotravel.add_objects(persons)

    # Public fluents
    person_at = Fluent('person_at', BoolType(), x=person, c=city)
    next = Fluent('next', BoolType(), l1=flevel, l2=flevel)
    zenotravel.ma_environment.add_fluent(person_at, default_initial_value=False)
    zenotravel.ma_environment.add_fluent(next, default_initial_value=False)

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

    for agent_name in instance['agents']:
        agent = Agent(agent_name, zenotravel)
        agent.add_fluent(fuel_level, default_initial_value=False)
        agent.add_fluent(carries, default_initial_value=False)
        agent.add_fluent(aircraft_at, default_initial_value=False)
        agent.add_action(board)
        agent.add_action(debark)
        agent.add_action(fly)
        agent.add_action(zoom)
        agent.add_action(refuel)
        zenotravel.add_agent(agent)

    for key in instance['init_values']:
        if key == 'global':
            for fluentuple in instance['init_values'][key]:
                fluent = zenotravel.ma_environment.fluent(fluentuple[0])
                params = (unified_planning.model.Object(v, obj_type[v]) for v in fluentuple[1])
                zenotravel.set_initial_value(fluent(*params), True)

        else:
            agent = zenotravel.agent(key)
            for fluentuple in instance['init_values'][key]:
                fluent = fluentuple[0]
                params = (unified_planning.model.Object(v, obj_type[v]) for v in fluentuple[1])
                zenotravel.set_initial_value(Dot(agent, agent.fluent(fluent)(*params)), True)

    agent_index = 0
    num_of_agents = len(zenotravel.agents)
    for goaltuple in instance['goals']:
        agent = zenotravel.agents[agent_index]
        agent_index = (agent_index + 1) % num_of_agents
        fluent = zenotravel.ma_environment.fluent(goaltuple[0])
        params = (unified_planning.model.Object(v, obj_type[v]) for v in goaltuple[1])
        agent.add_public_goal(fluent(*params))
    return zenotravel


def zenotravel_add_sociallaw(zenotravel):
    zenotravel_sl = SocialLaw()
    for agent in zenotravel.agents:
        zenotravel_sl.add_new_fluent(agent.name, 'assigned', (('p', 'person'),), False)
    persons_to_aircraft = {}
    for agent in zenotravel.agents:
        for goal in agent.public_goals:
            args = [arg.object() for arg in goal.args if arg.is_object_exp()]
            persons_args = [obj for obj in args if obj.type.name == 'person']
            for person in persons_args:
                persons_to_aircraft[person.name] = agent.name

    for person in zenotravel.objects(UserType('person')):
        if person.name in persons_to_aircraft:
            aircraft_name = persons_to_aircraft[person.name]
            zenotravel_sl.set_initial_value_for_new_fluent(aircraft_name, 'assigned', (person.name,), True)
    return zenotravel_sl.compile(zenotravel).problem


def get_new_slrc():
    return SocialLawRobustnessChecker(
        planner=planner,
        robustness_verifier_name="WaitingActionRobustnessVerifier")


def get_old_slrc():
    return SocialLawRobustnessChecker(
        planner=planner,
        robustness_verifier_name='SimpleInstantaneousActionRobustnessVerifier')


def check_robustness(slrc, problem):
    result = slrc.is_robust(problem)
    return str(result.status)


def get_intersection_problem(
        cars=["car-north", "car-south", "car-east", "car-west"],
        yields_list=[],
        wait_drive=True,
        durative=False) -> MultiAgentProblemWithWaitfor:
    # intersection multi agent
    problem = MultiAgentProblemWithWaitfor("intersection")

    loc = UserType("loc")
    direction = UserType("direction")
    car = UserType("car")

    # Environment
    connected = Fluent('connected', BoolType(), l1=loc, l2=loc, d=direction)
    free = Fluent('free', BoolType(), l=loc)
    if len(yields_list) > 0:
        yieldsto = Fluent('yieldsto', BoolType(), l1=loc, l2=loc)
        problem.ma_environment.add_fluent(yieldsto, default_initial_value=False)
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

    #  (:action arrive
    #     :agent    ?a - car
    #     :parameters  (?l - loc)
    #     :precondition  (and
    #     	(start ?a ?l)
    #     	(not (arrived ?a))
    #     	(free ?l)
    #       )
    #     :effect    (and
    #     	(at ?a ?l)
    #     	(not (free ?l))
    #     	(arrived ?a)
    #       )
    #   )
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

        #   (:action drive
    #     :agent    ?a - car
    #     :parameters  (?l1 - loc ?l2 - loc ?d - direction ?ly - loc)
    #     :precondition  (and
    #     	(at ?a ?l1)
    #     	(free ?l2)
    #     	(travel-direction ?a ?d)
    #     	(connected ?l1 ?l2 ?d)
    #     	(yields-to ?l1 ?ly)
    #     	(free ?ly)
    #       )
    #     :effect    (and
    #     	(at ?a ?l2)
    #     	(not (free ?l2))
    #     	(not (at ?a ?l1))
    #     	(free ?l1)
    #       )
    #    )
    # )
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
            drive.add_condition(StartTiming(), yieldsto(l1, ly))
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
            drive.add_precondition(yieldsto(l1, ly))
            drive.add_precondition(free(ly))
        drive.add_effect(at(l2), True)
        drive.add_effect(free(l2), False)
        drive.add_effect(at(l1), False)

        drive.add_effect(free(l1), True)

    plan = up.plans.SequentialPlan([])

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
                    problem.set_initial_value(yieldsto(problem.object(l1_name), problem.object(ly_name)), True)
                    yields.add(problem.object(l1_name))
                for l1 in problem.objects(loc):
                    if l1 not in yields:
                        problem.set_initial_value(yieldsto(l1, dummy_loc), True)

                        # slobjexp1 = (ObjectExp(slobj)),
            # plan.actions.append(up.plans.ActionInstance(arrive, slobjexp1, car))

            # for i in range(1,len(l)):
            #     flname = l[i-1]
            #     tlname = l[i]
            #     flobj = unified_planning.model.Object(flname, loc)
            #     tlobj = unified_planning.model.Object(tlname, loc)
            #     plan.actions.append(up.plans.ActionInstance(drive, (ObjectExp(flobj), ObjectExp(tlobj), ObjectExp(dobj) ), car))

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


def intersection_problem_add_sl1(i_prob):
    l = SocialLaw()
    for agent in i_prob.agents:
        l.add_waitfor_annotation(agent.name, "drive", "free", ("l2",))

    res = l.compile(i_prob)
    return res.problem


def intersection_problem_add_sl3(i_prob):
    p_4cars_deadlock = intersection_problem_add_sl1(i_prob)
    l3 = SocialLaw()
    l3.add_new_fluent(None, "yieldsto", (("l1", "loc"), ("l2", "loc")), False)
    l3.add_new_object("dummy_loc", "loc")
    for loc1, loc2 in [("south-ent", "cross-ne"), ("north-ent", "cross-sw"), ("east-ent", "cross-nw"),
                       ("west-ent", "cross-se")]:
        l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc1, loc2), True)
    for loc in i_prob.objects(i_prob.user_type("loc")):
        if loc.name not in ["south-ent", "north-ent", "east-ent", "west-ent"]:
            l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc.name, "dummy_loc"), True)
    for agent in i_prob.agents:
        l3.add_parameter_to_action(agent.name, "drive", "ly", "loc")
        l3.add_precondition_to_action(agent.name, "drive", "yieldsto", ("l1", "ly"))
        l3.add_precondition_to_action(agent.name, "drive", "free", ("ly",))
        l3.add_waitfor_annotation(agent.name, "drive", "free", ("ly",))
    return l3.compile(p_4cars_deadlock).problem


# Function to limit memory and CPU for the process
def set_limits(memory_limit, cpu_limit):
    # Set maximum memory usage (bytes)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    # Set maximum CPU time (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))


# Wrapper function to execute the target function with resource limits
def run_with_limits(func, args, memory_limit, cpu_limit, timeout, result_queue):
    # Apply memory and CPU limits
    set_limits(memory_limit, cpu_limit)
    try:
        start_time = time.time()
        result = func(*args)
        elapsed_time = time.time() - start_time
        result_queue.put({"result": result, "time": elapsed_time})
    except MemoryError:
        result_queue.put({"error": "Memory limit exceeded"})
    except Exception as e:
        result_queue.put({"error": str(e)})


# Main function to run the target function with specified limits
def run_experiment(func, args=(), memory_limit=8_192_000_000, cpu_limit=1800, timeout=None,
                   metadata=("unknown", False, False)):
    filename, old_compilation, has_social_law = metadata

    log_file = "./logs/experiment_log_" + date.today().strftime("%b-%d-%Y") + ".csv"

    result_queue = Queue()
    process = Process(target=run_with_limits, args=(func, args, memory_limit, cpu_limit, timeout, result_queue))

    process.start()
    process.join(timeout)

    if process.is_alive():
        # Log the timeout with "-"
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["-", filename, old_compilation, has_social_law])

        process.terminate()
        process.join()
        return {"error": "Timeout reached", "elapsed_time": "-"}

    if not result_queue.empty():
        result = result_queue.get()
        actual_time = result.get("time", "-")  # Retrieve elapsed time or "-" if missing
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([actual_time, filename, old_compilation, has_social_law])
        return result
    else:
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["-", filename, old_compilation, has_social_law])
        return {"error": "No result (possibly killed due to resource limits)", "elapsed_time": "-"}


def run_experiments(problems):
    total_problems = len(problems)

    # problems = [random.choice(problems) for _ in range (3)]

    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    log_file = "./logs/experiment_log_" + date.today().strftime("%b-%d-%Y") + ".csv"

    headers = ['time', 'name', 'slrc_is_old', 'has_social_law']

    # Create or overwrite the CSV file with the specified headers
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f'0/{total_problems} done.')
    for i, (name, problem, has_social_law) in enumerate(problems):
        for slrc_is_old in [True, False]:
            try:
                if slrc_is_old:
                    slrc = get_old_slrc()
                else:
                    slrc = get_new_slrc()
                run_experiment(
                    func=check_robustness,
                    args=(slrc, problem),
                    memory_limit=8_192_000_000,  # 8 GB
                    cpu_limit=1800,  # 30 minutes CPU time
                    timeout=3600,  # 1 hour wall time
                    metadata=(name, slrc_is_old, has_social_law)
                )

            except:
                pass
            print(f'Problem '+name+' with '+('old' if slrc_is_old else 'new')+' compilation is done.')
        print(f'{i + 1}/{total_problems}')


def get_problems():
    blocksworld_names = ['9-0', '9-1', '9-2', '10-0', '10-1', '10-2', '11-0', '11-1', '11-2', '12-0', '12-1', '13-0',
                         '13-1', '14-0', '14-1', '15-0', '15-1', '16-1', '16-2', ]  # '17-0']
    zenotravel_names = [f'pfile{i}' for i in range(3, 24)]
    zenotravel_names.remove('pfile11')
    driverlog_names = [f'pfile{i}' for i in range(1, 21)]
    grid_names = [
        (2, 3, 2),
        (3, 3, 3),
        (3, 4, 4),
        (4, 4, 5),
        (4, 5, 6),
        (5, 5, 6),
        (5, 6, 7),
        (6, 6, 8),
        (6, 7, 8),
        (7, 7, 7),
        (7, 8, 9),
        (8, 2, 2),
        (8, 3, 3),
        (8, 4, 6),
        (8, 5, 7),
        (8, 6, 8),
        (8, 7, 10),
        (6, 8, 8),
        (5, 8, 8),
        (3, 5, 3)
    ]
    problems = []
    driverlog_problems = []
    for name in driverlog_names:
        driverlog_problems.append((f'blocksworld_{name}', get_driverlog(name), False))
    problems += driverlog_problems
    blocksworld_problems = [(f'blocksworld_{name}', get_blocksworld(name), False) for name in blocksworld_names]
    problems += blocksworld_problems
    zenotravel_problems = [(f'zenotravel_{name}', get_zenotravel(name), False) for name in zenotravel_names]
    problems += zenotravel_problems
    zenotravel_problems_with_SL = [(f'zenotravel_sl_{name}', zenotravel_add_sociallaw(get_zenotravel(name)), True) for
                                   name in zenotravel_names]
    problems += zenotravel_problems_with_SL
    grid_problems = []
    grid_problems_with_SL = []

    for i, name in enumerate(grid_names):
        gm = GridManager(*name)
        gm.init_locs = INIT_LOCS[i]
        gm.goal_locs = GOAL_LOCS[i]
        p = gm.get_grid_problem()
        grid_problems.append(('grid_' + str(name).replace(' ', '_').replace(',', ''), p, False,))
        grid_problems_with_SL.append((f'grid_sl_{name}', gm.add_direction_law(p), True))
    problems += grid_problems
    problems += grid_problems_with_SL
    return problems


def read_data():
    exp_df = pandas.read_csv(
        '/home/evgeny/SocialLaws/up-social-laws/experimentation/logs/experiment_log_Oct-30-2024.csv')
    blocksworld_df = exp_df[exp_df.apply(lambda x: 'blocksworld' in x['name'], axis=1)]
    print('\nBLOCKSWORLD\n')
    print(blocksworld_df)
    zenotravel_sl_df = exp_df[exp_df.apply(lambda x: 'zenotravel' in x['name'] and 'sl' in x['name'], axis=1)]
    print('\nZENOTRAVEL_SL\n')
    print(zenotravel_sl_df)
    zenotravel_df = exp_df[exp_df.apply(lambda x: 'zenotravel' in x['name'] and not 'sl' in x['name'], axis=1)]
    print('\nZENOTRAVEL\n')
    print(zenotravel_df)
    driverlog_df = exp_df[exp_df.apply(lambda x: 'driverlog' in x['name'], axis=1)]
    print('\nDRIVERLOG\n')
    print(driverlog_df)
    grid_df = exp_df[exp_df.apply(lambda x: 'grid' in x['name'] and not 'sl' in x['name'], axis=1)]
    print('\nGRID\n')
    print(grid_df)
    grid_sl_df = exp_df[exp_df.apply(lambda x: 'grid' in x['name'] and 'sl' in x['name'], axis=1)]
    print('\nGRID_SL\n')
    print(grid_sl_df)


def transform_data(df):
    df = df.sort_values(by='name').reset_index(drop=True).drop('has_social_law', axis=1)
    slrc_old = df[df['slrc_is_old'] == True]
    slrc_old = slrc_old.rename(columns={'time': 'old_compilation'}, ).drop('slrc_is_old', axis=1)
    slrc_new = df[df['slrc_is_old'] == False]
    slrc_new = slrc_new.rename(columns={'time': 'new_compilation'}, ).drop('slrc_is_old', axis=1)
    df = pandas.merge(slrc_old, slrc_new, on='name', how='inner')
    df = df[['name', 'old_compilation', 'new_compilation']]
    return df


if __name__ == '__main__':

    #print(check_robustness(get_new_slrc(), problem))
    #print(check_robustness(get_new_slrc(), sl_problem))
    problems = []
    for num_of_agents in range(2, 4):
        gm = GridManager(3, 3, num_of_agents)
        problem = gm.get_grid_problem()
        sl_problem = gm.add_direction_law(problem)
        problems +=[(f'grid_problem_SL_{num_of_agents}', sl_problem, True),
                    (f'grid_problem_{num_of_agents}', problem, False)]
    run_experiments(problems)

    print(transform_data(pandas.read_csv('/home/evgeny/SocialLaws/up-social-laws/test/logs/experiment_log_Oct-31-2024.csv')))
