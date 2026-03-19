import datetime
import unified_planning
from unified_planning.shortcuts import *
import random

from experimentation.problem_generators.expedition_generator import ExpeditionGenerator
from experimentation.problem_generators.market_trader_generator import MarketTraderGenerator
from experimentation.problem_generators.numeric_grid_generator import NumericGridGenerator
from experimentation.problem_generators.numeric_zenotravel_generator import NumericZenotravelGenerator
from experimentation.problem_generators.numeric_civ_generator import NumericCivGenerator

from up_social_laws.ma_centralizer import MultiAgentProblemCentralizer
from up_social_laws.single_agent_projection import SingleAgentProjection
import os
import csv
import time
from datetime import date
from multiprocessing import Process, Queue
import resource


from up_social_laws.robustness_checker import SocialLawRobustnessChecker
from up_social_laws.snp_to_num_strips import MultiAgentNumericStripsProblemConverter, \
    MultiAgentWithWaitforNumericStripsProblemConverter


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
        name=get_general_slrc()._robustness_verifier_name,
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
            if print_state:
                print(state_to_str(state))

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

def get_general_slrc():
    slrc = SocialLawRobustnessChecker(
        planner=None,
        robustness_verifier_name="WaitingActionRobustnessVerifier")
    slrc.skip_checks = True
    return slrc

def get_simple_numeric_slrc():
    return SocialLawRobustnessChecker(
        planner=None,
        robustness_verifier_name='SimpleNumericRobustnessVerifier')

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
    def __init__(self, problems=None, engine='enhsp'):
        if problems is None:
            problems = []
        self.problems = problems
        self.mem_lim = 16_000_000_000  # 8 GB
        self.cpu_lim = 1800  # 30 minutes
        self.timeout = 3600  # 30 seconds timeout
        self.slrc = get_general_slrc()
        self.slrc.skip_checks = True
        self.slrc._planner = OneshotPlanner(name=engine)
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

    def debug_run_once(self, problem):
        return self.func(problem)

    def debug_run_full(self, problem):
        for prob in self.problems:
            self.debug_run_once(prob)

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
                      sl_options=(True, False), use_snp_converter=False):
        if probs is None:
            probs = range(1, 21)
        filepaths = {
            'grid': './numeric_problems_instances/grid/json',
            'zenotravel': './numeric_problems_instances/zenotravel/json',
            'expedition': './numeric_problems_instances/expedition/json',
            'markettrader': './numeric_problems_instances/markettrader/generated_json',
            'civ': './numeric_problems_instances/civ/json',
        }

        pgs = {
            'grid': NumericGridGenerator,
            'zenotravel': NumericZenotravelGenerator,
            'expedition': ExpeditionGenerator,
            'markettrader': MarketTraderGenerator,
            'civ': NumericCivGenerator
        }
        for prob_i in probs:
            for domain in domains:
                pg = pgs[domain]()
                pg.instances_folder = filepaths[domain]
                if domain in ['grid', 'zenotravel', 'expedition', 'civ']:
                    sl_opts = sl_options
                else:
                    sl_opts = [False, ]
                for has_sl in sl_opts:
                    prob = pg.generate_problem(f'pfile{prob_i}.json', sl=has_sl)
                    if use_snp_converter:
                        prob = MultiAgentWithWaitforNumericStripsProblemConverter(prob).compile()
                    self.problems.append((prob.name, prob, has_sl))
                    print(f'{prob.name} loaded')

def run_experiments():
    def prompt_choice(options, prompt="Enter choice: "):
        for i, opt in enumerate(options):
            print(f"  {i}: {opt}")
        while True:
            try:
                choice = int(input(prompt))
                if choice in range(len(options)):
                    return choice
            except ValueError:
                pass
            print("Invalid input. Try again.")

    confs = [
        (('expedition',), range(1, 21)),
        (('grid',), range(1, 21)),
        (('zenotravel',), range(1, 21)),
        (('markettrader',), range(1, 21)),
        (('civ',), range(1, 6)),
        (('grid', 'zenotravel', 'expedition', 'markettrader'), range(1, 21)),
    ]
    conf_labels = [
        "Expedition",
        "Grid",
        "Zenotravel",
        "Market Trader",
        "Civilization",
        "All domains",
    ]

    # --- Choose robustness checker ---
    print("\n=== Robustness Checker ===")
    checker_choice = prompt_choice(
        ["General (WaitingActionRobustnessVerifier)", "Simple Numeric (SimpleNumericRobustnessVerifier)", "Both (run general then simple)"],
        "Checker: "
    )

    # --- Choose social law ---
    print("\n=== Social Law ===")
    sl_choice = prompt_choice(
        ["With social law only", "Without social law only", "Both (with and without)"],
        "Social law: "
    )
    sl_options = {
        0: (True,),
        1: (False,),
        2: (True, False),
    }[sl_choice]

    # --- Choose configuration ---
    print("\n=== Configuration (domain + problem range) ===")
    conf_i = prompt_choice(conf_labels, "Conf: ")
    domains, probs = confs[conf_i]

    # --- Choose mode ---
    print("\n=== Mode ===")
    mode = prompt_choice(["Normal", "Debug"], "Mode: ")
    debug = (mode == 1)

    # --- Build experimentator ---
    exp = Experimentator(engine='tamer')

    # Apply chosen robustness checker
    if checker_choice == 0:
        slrc = get_general_slrc()
    else:
        slrc = get_simple_numeric_slrc()
    slrc.skip_checks = True
    slrc._planner = OneshotPlanner(name='tamer')
    exp.slrc = slrc
    exp.func = lambda p: check_robustness(exp.slrc, p)

    # If both checkers selected, load problems unconverted and convert per-checker when needed
    use_snp_for_load = (checker_choice == 1)
    if checker_choice == 2:
        use_snp_for_load = False
    exp.load_problems(domains=domains, probs=probs, sl_options=sl_options, use_snp_converter=use_snp_for_load)

    # Prepare list of checkers to run: tuple(slrc, needs_snp_conversion, label)
    if checker_choice == 2:
        checkers = [
            (get_general_slrc(), False, 'general'),
            (get_simple_numeric_slrc(), True, 'simple')
        ]
        exp.id = f"conf{conf_i}_both_sl{sl_choice}"
    else:
        if checker_choice == 0:
            checkers = [(get_general_slrc(), False, 'general')]
            exp.id = f"conf{conf_i}_general_sl{sl_choice}"
        else:
            checkers = [(get_simple_numeric_slrc(), True, 'simple')]
            exp.id = f"conf{conf_i}_simple_sl{sl_choice}"

    if debug:
        # --- Scope ---
        print("\n=== Scope ===")
        scope = prompt_choice(["Run all", "Single problem"], "Scope: ")

        def run_checkers_on_problem(name, problem):
            results = {}
            for slrc, needs_snp, label in checkers:
                print(f"\n=== Checker: {label} ===")
                slrc.skip_checks = True
                slrc._planner = OneshotPlanner(name='tamer')
                exp.slrc = slrc
                exp.func = lambda p, slrc=slrc: check_robustness(slrc, p)
                prob_to_use = problem
                if needs_snp:
                    try:
                        prob_to_use = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
                    except Exception as e:
                        print(f"Conversion failed for {name} under {label}: {e}")
                        results[label] = f"Conversion failed: {e}"
                        continue
                try:
                    res = exp.debug_run_once(prob_to_use)
                except Exception as e:
                    res = f"Error: {e}"
                print(f"Result ({label}): {res}")
                results[label] = res

            # If we have results from multiple checkers, compare equality
            if len(results) > 1:
                vals = list(results.values())
                try:
                    equal = all(v == vals[0] for v in vals)
                except Exception:
                    equal = False
                print("Comparison equal:", equal)
            return results

        if scope == 1:
            print(f"\n=== Problem (0–{len(exp.problems) - 1}) ===")
            for i, (name, _, has_sl) in enumerate(exp.problems):
                print(f"  {i}: {name} ({'with' if has_sl else 'without'} SL)")
            problem_id = prompt_choice([name for name, _, _ in exp.problems], "Problem ID: ")
            name, problem, has_sl = exp.problems[problem_id]
            run_checkers_on_problem(name, problem)
        else:
            for name, problem, has_sl in exp.problems:
                print(f"--- {name} ({'with' if has_sl else 'without'} SL) ---")
                run_checkers_on_problem(name, problem)

    else:
        # --- Choose single problem or all ---
        print("\n=== Scope ===")
        scope = prompt_choice(["Run all", "Single problem"], "Scope: ")

        print("\n=== Action ===")
        action = prompt_choice([
            "Run robustness check (default)",
            "Compile only",
            "Compile then run",
        ], "Action: ")

        if scope == 0:
            if action == 0:
                for slrc, needs_snp, label in checkers:
                    print(f"\n=== Running checker: {label} ===")
                    slrc.skip_checks = True
                    slrc._planner = OneshotPlanner(name='tamer')
                    exp.slrc = slrc
                    exp.func = lambda p, slrc=slrc: check_robustness(slrc, p)
                    if not needs_snp:
                        exp.experiment_full()
                    else:
                        # For SNP-needed checker, convert per-problem and run
                        exp.file_path = exp.log_dir + '/' + f"exp_log_{date.today().strftime('%b-%d-%Y')}_{exp.id}_{label}.csv"
                        with open(exp.file_path, mode="w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(['time', 'name', 'has_social_law', 'result'])
                        for name, problem, has_sl in exp.problems:
                            try:
                                conv = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
                                exp.experiment_once(conv, metadata=(name, has_sl))
                            except Exception as e:
                                print(f"Conversion/run failed for {name} under {label}: {e}")
            elif action == 1:
                for slrc, needs_snp, label in checkers:
                    print(f"\n=== Running checker (compile only): {label} ===")
                    slrc.skip_checks = True
                    slrc._planner = OneshotPlanner(name='tamer')
                    exp.slrc = slrc
                    exp.func = lambda p, slrc=slrc: check_robustness(slrc, p)
                    for name, problem, has_sl in exp.problems:
                        prob_to_use = problem
                        if needs_snp:
                            try:
                                prob_to_use = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
                            except Exception as e:
                                print(f"Conversion failed for {name} under {label}: {e}")
                                continue
                        try:
                            compiled = get_compiled_problem(prob_to_use)
                            print(f"Compiled problem for {name} (checker={label}): {compiled.name}")
                        except Exception as e:
                            print(f"Compilation failed for {name} under {label}: {e}")
            elif action == 2:
                for slrc, needs_snp, label in checkers:
                    print(f"\n=== Running checker (compile then run): {label} ===")
                    slrc.skip_checks = True
                    slrc._planner = OneshotPlanner(name='tamer')
                    exp.slrc = slrc
                    exp.func = lambda p, slrc=slrc: check_robustness(slrc, p)
                    exp.file_path = exp.log_dir + '/' + f"exp_log_{date.today().strftime('%b-%d-%Y')}_{exp.id}_{label}.csv"
                    with open(exp.file_path, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(['time', 'name', 'has_social_law', 'result'])
                    for name, problem, has_sl in exp.problems:
                        prob_to_use = problem
                        if needs_snp:
                            try:
                                prob_to_use = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
                            except Exception as e:
                                print(f"Conversion failed for {name} under {label}: {e}")
                                continue
                        try:
                            compiled = get_compiled_problem(prob_to_use)
                            print(f"Running compiled problem for {name} (checker={label})")
                            exp.experiment_once(compiled, metadata=(name, has_sl))
                        except Exception as e:
                            print(f"Compilation/run failed for {name} under {label}: {e}")
        else:
            print(f"\n=== Problem (0–{len(exp.problems) - 1}) ===")
            for i, (name, _, has_sl) in enumerate(exp.problems):
                print(f"  {i}: {name} ({'with' if has_sl else 'without'} SL)")
            problem_id = prompt_choice([name for name, _, _ in exp.problems], "Problem ID: ")
            name, problem, has_sl = exp.problems[problem_id]
            for slrc, needs_snp, label in checkers:
                print(f"\n=== Checker: {label} ===")
                slrc.skip_checks = True
                slrc._planner = OneshotPlanner(name='tamer')
                exp.slrc = slrc
                exp.func = lambda p, slrc=slrc: check_robustness(slrc, p)
                prob_to_use = problem
                if needs_snp:
                    try:
                        prob_to_use = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
                    except Exception as e:
                        print(f"Conversion failed for {name} under {label}: {e}")
                        continue
                if action == 0:
                    exp.experiment_once(prob_to_use, metadata=(name, has_sl))
                elif action == 1:
                    try:
                        compiled = get_compiled_problem(prob_to_use)
                        print(f"Compiled problem for {name} (checker={label}): {compiled.name}")
                    except Exception as e:
                        print(f"Compilation failed for {name} under {label}: {e}")
                elif action == 2:
                    try:
                        compiled = get_compiled_problem(prob_to_use)
                        exp.experiment_once(compiled, metadata=(name, has_sl))
                    except Exception as e:
                        print(f"Compilation/run failed for {name} under {label}: {e}")


if __name__ == '__main__':
    run_experiments()
