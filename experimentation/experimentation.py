import csv
import datetime
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
import resource
import time
import signal
import os
from multiprocessing import Process, Queue


# planner = OneshotPlanner()


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
    rbv.skip_checks = True

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


def simulate(problem, ma=False, print_state=False, trace_vars=[], random_walk=False):
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
                    if random_walk:
                        choice = random.choice(range(len(actions)))
                    else:
                        try:
                            choice = int(input('Choice: '))
                        except:
                            continue
                    if choice not in range(len(actions)):
                        print('Invalid index. Try again.')
                        continue
                    action = actions[int(choice)]
                    state = simulator.apply(state, action[0], action[1])
                    break
            except Exception as e:
                print(f'Applying action resulted in: \n{e}')
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
            for i, a in enumerate(actions):
                print(str(i) + '.', a[0].name, a[1])


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


def centralise(problem):
    mac = MultiAgentProblemCentralizer()
    mac.skip_checks = True
    return mac.compile(problem).problem


def get_new_slrc():
    slrc = SocialLawRobustnessChecker(
        planner=None,
        robustness_verifier_name="WaitingActionRobustnessVerifier")
    slrc.skip_checks = True
    return slrc


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
    #         (start ?a ?l)
    #         (not (arrived ?a))
    #         (free ?l)
    #       )
    #     :effect    (and
    #         (at ?a ?l)
    #         (not (free ?l))
    #         (arrived ?a)
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
    #         (at ?a ?l1)
    #         (free ?l2)
    #         (travel-direction ?a ?d)
    #         (connected ?l1 ?l2 ?d)
    #         (yields-to ?l1 ?ly)
    #         (free ?ly)
    #       )
    #     :effect    (and
    #         (at ?a ?l2)
    #         (not (free ?l2))
    #         (not (at ?a ?l1))
    #         (free ?l1)
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


import os
import csv
import time
from datetime import date
from multiprocessing import Process, Queue
import resource


# Function to set resource limits
def set_limits(memory_limit, cpu_limit):
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))  # Set max memory (bytes)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))  # Set max CPU time (seconds)


# Wrapper to execute a function with resource limits
def run_with_limits(func, args, memory_limit, cpu_limit, result_queue):
    try:
        set_limits(memory_limit, cpu_limit)  # Apply resource limits
        start_time = time.time()
        result = func(*args)
        elapsed_time = time.time() - start_time
        result_queue.put({"result": result, "time": elapsed_time})
    except MemoryError:
        result_queue.put({"error": "Memory limit exceeded"})
    except Exception as e:
        result_queue.put({"error": str(e)})


# Main Experimentator class
class Experimentator:
    def __init__(self, problems=[]):
        self.problems = problems
        self.mem_lim = 16_000_000_000  # 8 GB
        self.cpu_lim = 1800  # 30 minutes
        self.timeout = 3600  # 30 seconds timeout
        self.slrc = get_new_slrc()  # Assuming this function initializes the required object
        self.slrc.skip_checks = True
        self.slrc._planner = OneshotPlanner(name='enhsp')
        self.func = lambda p: check_robustness(self.slrc, p)  # Function to be executed

        self.log_dir = './logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_path = f"./logs/experiment_log_{date.today().strftime('%b-%d-%Y')}.csv"

    def experiment_once(self, problem, metadata=("unknown", False), ):
        filename, has_social_law = metadata
        result_queue = Queue()

        # Spawn a separate process to run the experiment
        process = Process(target=run_with_limits,
                          args=(self.func, (problem,), self.mem_lim, self.cpu_lim, result_queue))
        process.start()
        process.join(self.timeout)

        with open(self.file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if process.is_alive():
                # Timeout: terminate the process and log
                process.terminate()
                process.join()
                writer.writerow(["-", filename, has_social_law, 'timeout'])
                return {"error": "Timeout reached", "elapsed_time": "-"}

            if not result_queue.empty():
                result = result_queue.get()
                elapsed_time = result.get("time", "-")
                writer.writerow([elapsed_time, filename, has_social_law, result['result']])
                return result
            else:
                writer.writerow(["-", filename, has_social_law, 'NO_RES_ERR'])
                return {"error": "No result (possibly killed due to resource limits)", "elapsed_time": "-"}

    def experiment_full(self):
        total_problems = len(self.problems)
        headers = ['time', 'name', 'has_social_law', 'result']

        # Write headers to the log file
        with open(self.file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

        print(f'0/{total_problems} done.')
        for i, (name, problem, has_social_law) in enumerate(self.problems):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'Starting checking robustness of: {name} at time: {formatted_time}')
            try:
                result = self.experiment_once(problem, metadata=(name, has_social_law))
                print(f'Problem "{name}" ' + (
                    'with' if has_social_law else 'without') + f' social law is done with result:\n{result}')
            except Exception as e:
                print(f'Error while checking robustness of "{name}": {e}')
            # Format the time (optional)
            print(f'{i + 1}/{total_problems} done\n')

    def load_problems(self):
        filepaths = {
            'grid': './numeric_problems/grid/json',
            'zenotravel': './numeric_problems/zenotravel/json',
            'expedition': './numeric_problems/expedition/json',
            'markettrader': './numeric_problems/markettrader/generated_json',
        }

        pgs = {
            'grid': ProblemGenerator.NumericGridGenerator,
            'zenotravel': ProblemGenerator.NumericZenotravelGenerator,
            'expedition': ProblemGenerator.ExpeditionGenerator,
            'markettrader': ProblemGenerator.MarketTraderGenerator
        }

        domains = [
                'grid',
            #    'zenotravel',
            #    'expedition',
            #    'markettrader'
        ]

        for prob_i in range(1, 21):
            for domain in domains:
                pg = pgs[domain]()
                pg.instances_folder = filepaths[domain]
                if domain in ['grid', 'zenotravel', 'expedition']:
                    sl_options = [False, True]
                else:
                    sl_options = [False, ]
                for has_sl in sl_options:
                    prob = pg.generate_problem(f'pfile{prob_i}.json', sl=has_sl)
                    self.problems.append((prob.name, prob, has_sl))
                    print(f'{prob.name} loaded')


if __name__ == '__main__':
    exp = Experimentator()
    exp.load_problems()
    #prob = exp.problems[0][1]
    # sap = SingleAgentProjection(prob.agents[0])
    # sap.skip_checks = True
    # print(prob)
    # sap_prob = sap.compile(prob).problem
    # comp = exp.slrc.get_compiled(prob)
    # print(sap_prob)
    # simulate(comp)
    # print(comp)
    #print(OneshotPlanner(name='enhsp').solve(prob))
    #print(check_robustness(exp.slrc, prob))
    if input('run all exps?').lower() in ['y', 'yes', 'ok']:
        exp.experiment_full()
