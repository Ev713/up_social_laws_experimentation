import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Equals, And, Not, IntType, GE, LE, Minus, Plus

from experimentation.problem_generators.problem_generator import ProblemGenerator, NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


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

            agent.add_action(leave)

            agent.add_public_goal(self.fluent['left'][agent.name]())

        self.set_init_values()
        if sl:
            self.add_social_law()
        return self.problem
