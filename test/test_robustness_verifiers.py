from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import RealType, BoolType, Not, GE, OneshotPlanner

from experimentation.experimentator import simulate_problem
from experimentation.problem_generators import problem_generator, expedition_generator
from up_social_laws import snp_to_num_strips
from up_social_laws.SimpleNumericRobustnessVerifier import SimpleNumericRobustnessVerifier
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.robustness_checker import SocialLawRobustnessChecker
from up_social_laws.single_agent_projection import SingleAgentProjection
from up_social_laws.social_law import SocialLaw


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

if __name__ == '__main__':
    problem = gen_problem_with_social_law()
    #print(problem)
    #numeric_strips_problem = snp_to_num_strips.MultiAgentNumericStripsProblemConverter(problem).compile()
    slrc = SocialLawRobustnessChecker(robustness_verifier_name='SimpleNumericRobustnessVerifier')
    rv = SimpleNumericRobustnessVerifier()
    compiled = rv.compile(problem).problem
    sa_problem = SingleAgentProjection(problem.agents[0]).compile(problem).problem
    #print(compiled)
    print(sa_problem)
    #simulate_problem(compiled, print_state=True)

    with OneshotPlanner(problem_kind=sa_problem.kind) as planner:
        planning_result = planner.solve(sa_problem)
        print(planning_result)
