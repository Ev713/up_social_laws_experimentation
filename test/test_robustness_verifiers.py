from io import StringIO
from pathlib import Path

from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import RealType, BoolType, Not, GE, OneshotPlanner

from experimentation.problem_generators import problem_generator, expedition_generator
from experimentation.problem_generators.expedition_generator import ExpeditionGenerator
from experimentation.problem_generators.grid_generator import GridGenerator
from experimentation.problem_generators.market_trader_generator import MarketTraderGenerator
from experimentation.problem_generators.numeric_grid_generator import NumericGridGenerator
from experimentation.problem_generators.numeric_civ_generator import NumericCivGenerator
from experimentation.problem_generators.numeric_zenotravel_generator import NumericZenotravelGenerator

from up_social_laws import snp_to_num_strips
from up_social_laws.robustness_verification import SimpleNumericRobustnessVerifier
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.robustness_checker import SocialLawRobustnessChecker
from up_social_laws.single_agent_projection import SingleAgentProjection
from up_social_laws.snp_to_num_strips import NumericStripsProblemConverter, MultiAgentNumericStripsProblemConverter, MultiAgentWithWaitforNumericStripsProblemConverter
from up_social_laws.social_law import SocialLaw

def get_expedition_problem(sl=True):
    exp = ExpeditionGenerator()
    return exp.generate_problem('/home/ym/Documents/GitHub/up_social_laws_experimentation/experimentation/instances/numeric_expedition/minimal.json', sl)

def get_markettrader_problem(sl=True):
    pg = MarketTraderGenerator()
    return pg.generate_problem('/home/ym/Documents/GitHub/up_social_laws_experimentation/experimentation/instances/numeric_markettrader/pfile1.json', sl)

def get_grid_problem(sl=True):
    pg = NumericGridGenerator()
    return pg.generate_problem('/home/ym/Documents/GitHub/up_social_laws_experimentation/experimentation/instances/numeric_grid/pfile1.json', sl)

def get_civ_problem(sl=True):
    pg = NumericCivGenerator()
    return pg.generate_problem('/home/ym/Documents/GitHub/up_social_laws_experimentation/experimentation/instances/numeric_civ/pfile1.json', sl)

def get_numeric_zenotravel_problem(sl=True):
    pg = NumericZenotravelGenerator()
    return pg.generate_problem('/home/ym/Documents/GitHub/up_social_laws_experimentation/experimentation/instances/numeric_zenotravel/pfile1.json', sl)

def gen_problem_with_social_law():
    problem = MultiAgentProblemWithWaitfor('simple_numeric_tool_problem')
    N = 2
    agents = []
    for i in range(1, N+1):
        agent = Agent(name=f'agent{i}', ma_problem=problem)
        agents.append(agent)

    num_of_tools = Fluent('num_of_tools', RealType())
    tool_place_is_free = Fluent('tool_place_is_free', BoolType())
    problem.ma_environment.add_fluent(tool_place_is_free, default_initial_value=True)
    problem.ma_environment.add_fluent(num_of_tools, default_initial_value=len(agents)-1)

    at_tool_place = Fluent('at_tool_place', BoolType())
    at_home = Fluent('at_home', BoolType())
    has_problem = Fluent('has_problem', BoolType())
    has_tool = Fluent('has_tool', BoolType())
    has_no_tool = Fluent('has_no_tool', BoolType())

    go_to_tool_place = InstantaneousAction('go_to_tool_place',)
    go_to_tool_place.add_precondition(at_home)
    go_to_tool_place.add_precondition(tool_place_is_free)
    go_to_tool_place.add_effect(tool_place_is_free, False)
    go_to_tool_place.add_effect(at_tool_place, True)
    go_to_tool_place.add_effect(at_home, False)

    go_home = InstantaneousAction('go_home',)
    go_home.add_precondition(at_tool_place)
    go_home.add_effect(tool_place_is_free, True)
    go_home.add_effect(at_tool_place, False)
    go_home.add_effect(at_home, True)

    take_tool = InstantaneousAction('take_tool')
    take_tool.add_precondition(at_tool_place)
    take_tool.add_precondition(GE(num_of_tools, 1))
    take_tool.add_effect(num_of_tools, num_of_tools - 1)
    take_tool.add_effect(has_tool, True)
    take_tool.add_effect(has_no_tool, False)

    return_tool = InstantaneousAction('return_tool')
    return_tool.add_precondition(has_tool)
    return_tool.add_effect(num_of_tools, num_of_tools + 1)
    return_tool.add_effect(has_tool, False)
    return_tool.add_effect(has_no_tool, True)

    fix_problem = InstantaneousAction('fix_problem')
    fix_problem.add_precondition(Not(at_tool_place))
    fix_problem.add_precondition(has_tool)
    fix_problem.add_effect(has_problem, False)

    for a in agents:
        a.add_fluent(at_tool_place, default_initial_value=False)
        a.add_fluent(has_problem, default_initial_value=True)
        a.add_fluent(at_home, default_initial_value=True)
        a.add_fluent(has_tool, default_initial_value=False)
        a.add_fluent(has_no_tool, default_initial_value=True)

        a.add_action(go_to_tool_place)
        a.add_action(go_home)
        a.add_action(take_tool)
        a.add_action(return_tool)
        a.add_action(fix_problem)

        a.add_public_goal(Not(has_problem))
        problem.add_agent(a)

    social_law = SocialLaw()
    for agent in agents:
        social_law.add_new_fluent(agent.name, f'needs_access', (), True)
        social_law.add_effect(agent.name, f'go_to_tool_place', 'needs_access', (), False)
        social_law.add_precondition_to_action(agent.name, 'go_to_tool_place', 'needs_access', ())
        social_law.add_waitfor_annotation(agent.name, 'go_to_tool_place', 'tool_place_is_free', ())
        social_law.add_waitfor_annotation(agent.name, 'go_to_tool_place', 'num_of_tools', (), '>=', 1)
        social_law.add_waitfor_annotation(agent.name, 'take_tool', 'num_of_tools', (), '>=', 1)
        social_law.add_agent_goal(agent.name, 'at_home', ())
        social_law.add_agent_goal(agent.name, 'has_no_tool', ())
    problem = social_law.compile(problem).problem

    return problem


def test_linear_effects_table_supports_assign_increase_decrease():
    problem = MultiAgentProblemWithWaitfor('numeric_effect_kinds')
    agent = Agent(name='agent1', ma_problem=problem)

    level = Fluent('level', RealType())
    agent.add_fluent(level, default_initial_value=0)

    assign = InstantaneousAction('assign')
    assign.add_effect(level, level + 3)

    increase = InstantaneousAction('increase')
    increase.add_increase_effect(level, 2)

    decrease = InstantaneousAction('decrease')
    decrease.add_decrease_effect(level, 4)

    agent.add_action(assign)
    agent.add_action(increase)
    agent.add_action(decrease)
    problem.add_agent(agent)

    table = MultiAgentNumericStripsProblemConverter(problem).create_linear_effects_table()
    changes = {row['action_id'][1]: row['change'] for _, row in table.iterrows()}

    assert changes['assign'] == 3
    assert changes['increase'] == 2
    assert changes['decrease'] == -4


def test_simple_numeric_robustness_copy_preserves_inc_dec_effect_kinds():
    problem = MultiAgentProblemWithWaitfor('numeric_effect_copy')
    agent = Agent(name='agent1', ma_problem=problem)

    level = Fluent('level', RealType())
    agent.add_fluent(level, default_initial_value=0)

    increase = InstantaneousAction('increase')
    increase.add_increase_effect(level, 2)

    decrease = InstantaneousAction('decrease')
    decrease.add_decrease_effect(level, 4)

    agent.add_action(increase)
    agent.add_action(decrease)
    problem.add_agent(agent)

    verifier = SimpleNumericRobustnessVerifier()
    verifier.initialize_problem(problem)

    copied_increase = verifier.create_action_copy(problem, agent, increase, 'copy')
    copied_decrease = verifier.create_action_copy(problem, agent, decrease, 'copy')

    assert copied_increase.effects[0].is_increase()
    assert copied_increase.effects[0].value.constant_value() == 2
    assert copied_decrease.effects[0].is_decrease()
    assert copied_decrease.effects[0].value.constant_value() == 4


def test_civ_integer_fluents_are_bounded():
    problem = get_civ_problem(sl=False)

    fluents = list(problem.ma_environment.fluents)
    for agent in problem.agents:
        fluents.extend(agent.fluents)

    for fluent in fluents:
        if fluent.type.is_int_type():
            assert fluent.type.lower_bound is not None
            assert fluent.type.upper_bound is not None


def test_markettrader_simple_numeric_compilation_works():
    problem = get_markettrader_problem(sl=False)
    snp_problem = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
    compiled = SimpleNumericRobustnessVerifier().compile(snp_problem).problem
    assert compiled is not None


def test_numeric_zenotravel_simple_numeric_compilation_works():
    for sl in (False, True):
        problem = get_numeric_zenotravel_problem(sl=sl)
        snp_problem = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
        compiled = SimpleNumericRobustnessVerifier().compile(snp_problem).problem
        assert compiled is not None


def test_numeric_grid_simple_numeric_compilation_works():
    for sl in (False, True):
        problem = get_grid_problem(sl=sl)
        snp_problem = MultiAgentWithWaitforNumericStripsProblemConverter(problem).compile()
        compiled = SimpleNumericRobustnessVerifier().compile(snp_problem).problem
        assert compiled is not None

def solve_with_log(problem, log_path, planner_name='enhsp', timeout=None):
    stream = StringIO()
    with OneshotPlanner(name=planner_name, problem_kind=problem.kind) as planner:
        result = planner.solve(problem, timeout=timeout, output_stream=stream)
    log_path.write_text(stream.getvalue())
    return result

def safe_log_name(name):
    return name.replace('/', '_').replace('\\', '_').replace(':', '_')

if __name__ == '__main__':
    logs_dir = Path(__file__).resolve().parent / ".logs"
    logs_dir.mkdir(exist_ok=True)

    ma_problem = get_civ_problem(sl=True)

    simple_problem = MultiAgentWithWaitforNumericStripsProblemConverter(ma_problem).compile()
    print(f"Simple numeric source problem: {simple_problem.name}")
    for agent in simple_problem.agents:
        sap = SingleAgentProjection(agent)
        sap.skip_checks = True
        sa_problem = sap.compile(simple_problem).problem
        sa_log_path = logs_dir / f"{safe_log_name(simple_problem.name)}_{agent.name}_single_agent.log"
        sa_result = solve_with_log(sa_problem, sa_log_path)
        print(f"Single-agent projection for {agent.name}: {sa_result.status}")
        print(f"  log: {sa_log_path}")

    simple_compiled = SimpleNumericRobustnessVerifier().compile(simple_problem).problem
    simple_log_path = logs_dir / f"{safe_log_name(simple_compiled.name)}_robustness.log"
    simple_result = solve_with_log(simple_compiled, simple_log_path)
    print(f"Simple robustness compilation: {simple_result.status}")
    print(f"  log: {simple_log_path}")

    general_checker = SocialLawRobustnessChecker(
        robustness_verifier_name='WaitingActionRobustnessVerifier'
    )
    general_checker.skip_checks = True
    general_compiled = general_checker.get_compiled(ma_problem)
    general_log_path = logs_dir / f"{safe_log_name(general_compiled.name)}_robustness.log"
    general_result = solve_with_log(general_compiled, general_log_path, timeout=30)
    print(f"General robustness compilation: {general_result.status}")
    print(f"  log: {general_log_path}")
