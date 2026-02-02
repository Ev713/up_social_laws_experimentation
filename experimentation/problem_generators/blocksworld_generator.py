from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor


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