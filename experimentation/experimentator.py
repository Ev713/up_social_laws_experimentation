import datetime
import unified_planning
from unified_planning.shortcuts import *
import random
from problem_generators.expedition_generator import ExpeditionGenerator
from problem_generators.market_trader_generator import MarketTraderGenerator
from problem_generators.numeric_grid_generator import NumericGridGenerator
from problem_generators.numeric_zenotravel_generator import NumericZenotravelGenerator

from up_social_laws.ma_centralizer import MultiAgentProblemCentralizer
from up_social_laws.single_agent_projection import SingleAgentProjection
import os
import csv
import time
from datetime import date
from multiprocessing import Process, Queue
import resource


from up_social_laws.robustness_checker import SocialLawRobustnessChecker


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

def simulate_problem(problem, ma=False, print_state=False, trace_vars=[], random_walk=False):
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

def set_limits(memory_limit, cpu_limit):
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))  # Set max memory (bytes)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))  # Set max CPU time (seconds)

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


class Experimentator:
    def __init__(self, problems=None):
        if problems is None:
            problems = []
        self.problems = problems
        self.mem_lim = 16_000_000_000  # 8 GB
        self.cpu_lim = 1800  # 30 minutes
        self.timeout = 3600  # 30 seconds timeout
        self.slrc = get_new_slrc()  # Assuming this function initializes the required object
        self.slrc.skip_checks = True
        self.slrc._planner = OneshotPlanner(name='enhsp')
        self.func = lambda p: check_robustness(self.slrc, p)  # Function to be executed
        self.id = '_'
        self.log_dir = '.archive/logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_path = ''

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

        self.file_path = self.log_dir + '/' + f"exp_log_{date.today().strftime('%b-%d-%Y')}_{self.id}.csv"
        print('Writing to:', self.file_path)

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

    def load_problems(self, domains=('grid', 'zenotravel', 'expedition', 'markettrader'), probs=None,
                      sl_options=(True, False)):
        if probs is None:
            probs = range(1, 21)
        filepaths = {
            'grid': './numeric_problems_instances/grid/json',
            'zenotravel': './numeric_problems_instances/zenotravel/json',
            'expedition': './numeric_problems_instances/expedition/json',
            'markettrader': './numeric_problems_instances/markettrader/generated_json',
        }

        pgs = {
            'grid': NumericGridGenerator,
            'zenotravel': NumericZenotravelGenerator,
            'expedition': ExpeditionGenerator,
            'markettrader': MarketTraderGenerator
        }
        for prob_i in probs:
            for domain in domains:
                pg = pgs[domain]()
                pg.instances_folder = filepaths[domain]
                if domain in ['grid', 'zenotravel', 'expedition']:
                    sl_opts = sl_options
                else:
                    sl_opts = [False, ]
                for has_sl in sl_opts:
                    prob = pg.generate_problem(f'pfile{prob_i}.json', sl=has_sl)
                    self.problems.append((prob.name, prob, has_sl))
                    print(f'{prob.name} loaded')


if __name__ == '__main__':
    debug = False
    exp = Experimentator()
    conf = [
        (('expedition',), range(1, 21), (True,)),
        (('expedition',), range(1, 21), (False,)),
        'Debug',
        'Exit'
    ]
    for i, conf_i in enumerate(conf):
        print(f'{i}: {conf_i}')
    bug = True
    while bug:
        bug = False
        try:
            i = int(input('Enter conf:'))
            conf_i = conf[i]
            if conf_i == 'Debug':
                debug = True
        except:
            print('Unreadable index. Try again.')
            bug = True
    if not debug:
        exp.load_problems(*conf_i)
        exp.id = str(i)
        if input('run all exps?').lower() in ['y', 'yes', 'ok']:
            exp.experiment_full()

