import json
import random

import unified_planning
from unified_planning.model import InstantaneousAction, Fluent
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import *

from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw
EXPERIMENTATION_PATH = r'C:\\Users\\foree\\PycharmProjects\\up_social_laws_experimentation\\experimentation\\'

def get_blocksworld(name):
    json_file_path = f'{EXPERIMENTATION_PATH}/problems/all/jsons/blocksworld/{name}.json'

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
    for agent in zenotravel.agents:
        zenotravel_sl.add_precondition_to_action(agent.name, 'board', 'assigned', ('p',))
    for person in zenotravel.objects(UserType('person')):
        if person.name in persons_to_aircraft:
            aircraft_name = persons_to_aircraft[person.name]
            zenotravel_sl.set_initial_value_for_new_fluent(aircraft_name, 'assigned', (person.name,), True)
    zenotravel_sl.skip_checks=True
    return zenotravel_sl.compile(zenotravel).problem


def get_zenotravel(name):
    filepath = open(
        f"{EXPERIMENTATION_PATH}/numeric_problems/all/jsons/zenotravel/{name}.json").read()
    instance = json.loads(filepath)
    zenotravel = MultiAgentProblemWithWaitfor()

    # Object types
    city = UserType('city')
    person = UserType('person')

    obj_type = {}
    for objname in instance['citys']:
        obj_type[objname] = city
    for objname in instance['persons']:
        obj_type[objname] = person

    citys = list(map(lambda c: unified_planning.model.Object(c, city), instance['citys']))
    zenotravel.add_objects(citys)
    persons = list(map(lambda p: unified_planning.model.Object(p, person), instance['persons']))
    zenotravel.add_objects(persons)

    # Public fluents
    person_loc = Fluent('person-loc', BoolType(), x=person, c=city)
    zenotravel.ma_environment.add_fluent(person_loc, default_initial_value=False)
    distance = Fluent('distance', RealType(), c1=city, c2=city)
    zenotravel.ma_environment.add_fluent(distance, default_initial_value=False)

    # Agent fluents
    carries = Fluent('carries', BoolType(), p=person)
    fuel = Fluent('fuel', RealType(),)
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
    board.add_effect(onboard, Plus(onboard+1))
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

    fly_fast = InstantaneousAction('fly-fast', c1=city, c2=city,)
    c1 = fly_fast.parameter('c1')
    c2 = fly_fast.parameter('c2')
    fly_fast.add_precondition(aircraft_loc(c1))
    fly_fast.add_precondition(GT(distance(c1, c2), 0))
    fly_fast.add_precondition(GE(fuel, Times(distance(c1, c2), fast_burn)))
    fly_fast.add_precondition(GE(zoom_limit, onboard))
    fly_fast.add_effect(aircraft_loc(c2), True)
    fly_fast.add_effect(aircraft_loc(c1), False)
    fly_fast.add_effect(fuel, Minus(fuel, Times(distance(c1, c2), fast_burn)))

    refuel = InstantaneousAction('refuel',)
    refuel.add_precondition(GT(capacity, fuel))
    refuel.add_effect(fuel, capacity)

    for agent_name in instance['agents']:
        agent = Agent(agent_name, zenotravel)
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
        zenotravel.add_agent(agent)

    for key in instance['init_values']:
        for fluentuple in instance['init_values'][key]:
            value = True
            if fluentuple[0] == '=':
                value = fluentuple[-1]
                fluentuple = fluentuple[1]

            if key == 'global':
                fluent = zenotravel.ma_environment.fluent(fluentuple[0])
                params = (unified_planning.model.Object(v, obj_type[v]) for v in fluentuple[1])
                zenotravel.set_initial_value(fluent(*params), value)
            else:
                agent = zenotravel.agent(key)
                fluent = agent.fluent(fluentuple[0])
                params = (unified_planning.model.Object(v, obj_type[v]) for v in fluentuple[1])
                agent = zenotravel.agent(key)
                zenotravel.set_initial_value(Dot(agent, fluent(*params)), value)
    agent_index = 0
    num_of_agents = len(zenotravel.agents)
    for goaltuple in instance['goals']:
        agent_fluent=False
        vars = goaltuple[1][1] if goaltuple[0] == '=' else goaltuple[1]
        for var in vars:
            if 'plane' in var:
                agent = zenotravel.agent(var)
                agent_fluent=True
                vars.remove(var)
        if not agent_fluent:
            agent = zenotravel.agents[agent_index]
            agent_index = (agent_index + 1) % num_of_agents
        if goaltuple[0] == '=':
            if agent_fluent:
                fluent = agent.fluent(goaltuple[1][0])
            else:
                fluent = zenotravel.ma_environment.fluent(goaltuple[1][0])
            params = (unified_planning.model.Object(v, obj_type[v]) for v in goaltuple[1][1])

        else:
            if agent_fluent:
                fluent = agent.fluent(goaltuple[0])
            else:
                fluent = zenotravel.ma_environment.fluent(goaltuple[0])
            params = (unified_planning.model.Object(v, obj_type[v]) for v in goaltuple[1])
        agent.add_public_goal(fluent(*params))
    return zenotravel


def get_numeric_problem():
    problem = MultiAgentProblemWithWaitfor()

    # Declaring types
    charger = UserType("charger")

    # Creating problem ‘variables’
    is_free = Fluent('is_free', BoolType(), c=charger)

    # Declaring objects
    charger1 = Object("charger1", charger)
    # Populating the problem with initial state and goals
    problem.ma_environment.add_fluent(is_free, default_initial_value=True)

    for agent in [Agent(f'robot_{i}', problem) for i in range(2)]:
        plugged_in = Fluent("plugged_in", BoolType(), c=charger)
        battery = Fluent("battery", RealType())

        charge = InstantaneousAction("charge", plugged_charger=charger)
        pc = charge.parameter("plugged_charger")
        charge.add_precondition(plugged_in(pc))
        charge.add_precondition(LE(battery, 100))
        charge.add_effect(plugged_in(pc), False)
        charge.add_effect(is_free(pc), True)
        charge.add_effect(battery, Plus(battery, 100))

        plug_in = InstantaneousAction("plug_in", free_charger=charger)
        fc = plug_in.parameter("free_charger")
        plug_in.add_precondition(is_free(fc))
        plug_in.add_effect(plugged_in(fc), True)
        plug_in.add_effect(is_free(fc), False)

        agent.add_action(charge)
        agent.add_action(plug_in)
        agent.add_fluent(battery, default_initial_value=0)
        agent.add_fluent(plugged_in, default_initial_value=False)
        agent.add_public_goal(Equals(battery, 100))
        problem.add_agent(agent)

    problem.add_object(charger1)
    return problem


def numeric_with_sl():
    sl = SocialLaw()
    sl.skip_checks = True
    numeric = get_numeric_problem()
    for agent in numeric.agents:
        sl.add_waitfor_annotation(agent.name, 'plug_in', 'is_free', ('free_charger',))
        sl.add_agent_complex_goal(agent.name,'NOT', ('plugged_in', ), (('charger1', ), ))
    return sl.compile(numeric).problem


def sa_numeric():
    # Declaring types
    Location = UserType("Location")

    # Creating problem ‘variables’
    robot_at = Fluent("robot_at", BoolType(), location=Location)
    battery_charge = Fluent("battery_charge", RealType())

    # Creating actions
    move = InstantaneousAction("move", l_from=Location, l_to=Location)
    l_from = move.parameter("l_from")
    l_to = move.parameter("l_to")
    move.add_precondition(GE(battery_charge, 10))
    move.add_precondition(robot_at(l_from))
    move.add_precondition(Not(robot_at(l_to)))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)
    move.add_effect(battery_charge, Minus(battery_charge, 10))

    # Declaring objects
    l1 = Object("l1", Location)
    l2 = Object("l2", Location)

    # Populating the problem with initial state and goals
    problem = Problem("robot")
    problem.add_fluent(robot_at)
    problem.add_fluent(battery_charge)
    problem.add_action(move)
    problem.add_object(l1)
    problem.add_object(l2)
    problem.set_initial_value(robot_at(l1), True)
    problem.set_initial_value(robot_at(l2), False)
    problem.set_initial_value(battery_charge, 100)
    problem.add_goal(robot_at(l2))
    return problem
