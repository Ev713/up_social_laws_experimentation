import csv
import json
from datetime import date

import pandas
import unified_planning
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
import random
from unified_planning.io import PDDLWriter, MAPDDLWriter
import ProblemGenerator
import problems
import numeric_problems

from up_social_laws.waiting_robustness_verification import RegularWaitingActionRobustnessVerifier

up.shortcuts.get_environment().credits_stream = None

# from test.test_social_law import Example
from up_social_laws.ma_centralizer import MultiAgentProblemCentralizer
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.robustness_verification import WaitingActionRobustnessVerifier
from up_social_laws.single_agent_projection import SingleAgentProjection
from up_social_laws.social_law import SocialLaw

from up_social_laws.robustness_checker import SocialLawRobustnessChecker
from unified_planning.io import PDDLReader
#import resource
import time
import signal
import os
from multiprocessing import Process, Queue

#planner = OneshotPlanner()


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
        sap.skip_checks = True
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
        for i, a in enumerate(actions):
            print(f'{i}: {a[0].name}{a[1]}')
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


def solve(problem, ma=False):
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
    mac.skip_checks = True
    return mac.compile(problem).problem


def get_new_slrc():
    return SocialLawRobustnessChecker(
        planner=None,
        robustness_verifier_name="WaitingActionRobustnessVerifier")


def get_old_slrc():
    return SocialLawRobustnessChecker(
        planner=None,
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
'''def set_limits(memory_limit, cpu_limit):
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


def run_experiments(problems, slrc_old_options=[True, False]):
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
        for slrc_is_old in slrc_old_options:
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
            print(f'Problem ' + name + ' with ' + ('old' if slrc_is_old else 'new') + ' compilation is done.')
        print(f'{i + 1}/{total_problems}')
'''
def get_problems():
    blocksworld_names = ['9-0', '9-1', '9-2', '10-0', '10-1', '10-2', '11-0', '11-1', '11-2', '12-0', '12-1', '13-0',
                         '13-1', '14-0', '14-1', '15-0', '15-1', '16-1', '16-2', ]  # '17-0']
    zenotravel_names = ['pfile3', 'pfile8']  # [f'pfile{i}' for i in range(3, 24)]
    if 'pfile11' in zenotravel_names:
        zenotravel_names.remove('pfile11')
    # driverlog_names = [f'pfile{i}' for i in range(1, 21)].
    driverlog_names = ['pfile1', 'pfile6', 'pfile7']
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
        driverlog_problems.append((f'driverlog_{name}', problems.get_driverlog(name), False))
    blocksworld_problems = [(f'blocksworld_{name}', problems.get_blocksworld(name), False) for name in
                            blocksworld_names]
    zenotravel_problems = [(f'zenotravel_{name}', problems.get_zenotravel(name), False) for name in zenotravel_names]
    zenotravel_problems_with_SL = [
        (f'zenotravel_SL_{name}', problems.zenotravel_add_sociallaw(problems.get_zenotravel(name)), True) for
        name in zenotravel_names]
    grid_problems = []
    grid_problems_with_SL = []

    for i, name in enumerate(grid_names):
        gm = problems.GridManager(*name)
        gm.init_locs = problems.INIT_LOCS[i]
        gm.goal_locs = problems.GOAL_LOCS[i]
        p = gm.get_grid_problem()
        grid_problems.append(('grid_' + str(name).replace(' ', '_').replace(',', ''), p, False,))
        grid_problems_with_SL.append((f'grid_SL_{name}', gm.add_direction_law(p), True))

    problems += driverlog_problems
    # problems += blocksworld_problems
    problems += zenotravel_problems
    # problems += zenotravel_problems_with_SL
    # problems += grid_problems
    # problems += grid_problems_with_SL
    return problems


if __name__ == '__main__':
    pg = ProblemGenerator.MarketTraderGenerator()
    pg.instances_folder = r'./numeric_problems/markettrader/json'
    problem = pg.generate_problem('pfile1.json', sl=False)
    comp = get_compiled_problem(problem)
    with OneshotPlanner(name='tamer', problem_kind=problem.kind) as planner:
        result = planner.solve(comp)
        print(result)
