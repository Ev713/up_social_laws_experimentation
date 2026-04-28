from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.shortcuts import BoolType, IntType, Not

from experimentation.problem_generators.problem_generator import NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor


class NumericGridGenerator(NumericProblemGenerator):

    def __init__(self):
        super().__init__()
        self.fluent = {
            'agent_x': {},
            'agent_y': {},
            'goal_x': {},
            'goal_y': {},
            'on_map': {},
            'left': {},
            'at': {},
            'free': {}
        }

    def _cell_key(self, x, y):
        return f'{x}_{y}'

    def _allowed_directions(self):
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

        allowed = {}
        min_x = self.instance_data['min_x']
        max_x = self.instance_data['max_x']
        min_y = self.instance_data['min_y']
        max_y = self.instance_data['max_y']
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cell_allowed = {'up': True, 'down': True, 'left': True, 'right': True}
                if y < max_y:
                    cell_allowed['right'] = False
                if y > min_y:
                    cell_allowed['left'] = False
                if x not in up_columns:
                    cell_allowed['up'] = False
                if x not in down_columns:
                    cell_allowed['down'] = False
                allowed[(x, y)] = cell_allowed
        return allowed

    def _destination(self, direction, x, y):
        return {
            'up': (x, y + 1),
            'down': (x, y - 1),
            'left': (x - 1, y),
            'right': (x + 1, y),
        }[direction]

    def _in_bounds(self, x, y):
        return self.instance_data['min_x'] <= x <= self.instance_data['max_x'] and \
            self.instance_data['min_y'] <= y <= self.instance_data['max_y']

    def _is_allowed(self, sl, allowed_directions, direction, x, y):
        if not self._in_bounds(x, y):
            return False
        if not sl:
            return True
        return allowed_directions[(x, y)][direction]

    def _create_cell_fluents(self):
        min_x = self.instance_data['min_x']
        max_x = self.instance_data['max_x']
        min_y = self.instance_data['min_y']
        max_y = self.instance_data['max_y']
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cell_key = self._cell_key(x, y)
                self.fluent['free'][cell_key] = Fluent(f'free_{cell_key}', BoolType())
                self.problem.ma_environment.add_fluent(self.fluent['free'][cell_key], default_initial_value=True)

    def _create_agent_fluents(self, agent):
        min_x = self.instance_data['min_x']
        max_x = self.instance_data['max_x']
        min_y = self.instance_data['min_y']
        max_y = self.instance_data['max_y']
        name = agent.name
        self.fluent['agent_x'][name] = Fluent(f'{name}_x', IntType(min_x, max_x))
        self.fluent['agent_y'][name] = Fluent(f'{name}_y', IntType(min_y, max_y))
        self.fluent['goal_x'][name] = Fluent(f'{name}_goal_x', IntType(min_x, max_x))
        self.fluent['goal_y'][name] = Fluent(f'{name}_goal_y', IntType(min_y, max_y))
        self.fluent['on_map'][name] = Fluent(f'{name}_on_map', BoolType())
        self.fluent['left'][name] = Fluent(f'{name}_left', BoolType())

        for key in ['agent_x', 'agent_y', 'goal_x', 'goal_y', 'on_map', 'left']:
            default = False if key in ['on_map', 'left'] else 0
            self.problem.ma_environment.add_fluent(self.fluent[key][name], default_initial_value=default)

        self.fluent['at'][name] = {}
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                cell_key = self._cell_key(x, y)
                self.fluent['at'][name][cell_key] = Fluent(f'{name}_at_{cell_key}', BoolType())
                self.problem.ma_environment.add_fluent(self.fluent['at'][name][cell_key], default_initial_value=False)

    def _set_agent_initial_values(self, agent_name):
        agent_data = self.instance_data[agent_name]
        init_x = agent_data['init_x']
        init_y = agent_data['init_y']
        goal_x = agent_data['goal_x']
        goal_y = agent_data['goal_y']

        self.problem.set_initial_value(self.fluent['agent_x'][agent_name](), init_x)
        self.problem.set_initial_value(self.fluent['agent_y'][agent_name](), init_y)
        self.problem.set_initial_value(self.fluent['goal_x'][agent_name](), goal_x)
        self.problem.set_initial_value(self.fluent['goal_y'][agent_name](), goal_y)
        self.problem.set_initial_value(self.fluent['on_map'][agent_name](), False)
        self.problem.set_initial_value(self.fluent['left'][agent_name](), False)

    def _add_appear_action(self, agent):
        name = agent.name
        init_x = self.instance_data[name]['init_x']
        init_y = self.instance_data[name]['init_y']
        init_key = self._cell_key(init_x, init_y)

        appear = InstantaneousAction('appear')
        appear.add_precondition(Not(self.fluent['on_map'][name]()))
        appear.add_precondition(Not(self.fluent['left'][name]()))
        appear.add_precondition(self.fluent['free'][init_key]())
        appear.add_effect(self.fluent['on_map'][name](), True)
        appear.add_effect(self.fluent['at'][name][init_key](), True)
        appear.add_effect(self.fluent['free'][init_key](), False)
        agent.add_action(appear)
        if getattr(self.problem, 'waitfor', None) is not None:
            self.problem.waitfor.annotate_as_waitfor(name, appear.name, self.fluent['free'][init_key]())

    def _add_leave_action(self, agent):
        name = agent.name
        goal_x = self.instance_data[name]['goal_x']
        goal_y = self.instance_data[name]['goal_y']
        goal_key = self._cell_key(goal_x, goal_y)

        leave = InstantaneousAction('leave')
        leave.add_precondition(self.fluent['on_map'][name]())
        leave.add_precondition(self.fluent['at'][name][goal_key]())
        leave.add_effect(self.fluent['left'][name](), True)
        leave.add_effect(self.fluent['on_map'][name](), False)
        leave.add_effect(self.fluent['at'][name][goal_key](), False)
        leave.add_effect(self.fluent['free'][goal_key](), True)
        agent.add_action(leave)

    def _add_move_actions(self, agent, sl, allowed_directions):
        name = agent.name
        min_x = self.instance_data['min_x']
        max_x = self.instance_data['max_x']
        min_y = self.instance_data['min_y']
        max_y = self.instance_data['max_y']

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                src_key = self._cell_key(x, y)
                for direction in ['up', 'down', 'left', 'right']:
                    if not self._is_allowed(sl, allowed_directions, direction, x, y):
                        continue
                    dst_x, dst_y = self._destination(direction, x, y)
                    if not self._in_bounds(dst_x, dst_y):
                        continue
                    dst_key = self._cell_key(dst_x, dst_y)
                    action = InstantaneousAction(f'move_{direction}_{src_key}')
                    action.add_precondition(self.fluent['on_map'][name]())
                    action.add_precondition(self.fluent['at'][name][src_key]())
                    action.add_precondition(self.fluent['free'][dst_key]())
                    action.add_effect(self.fluent['at'][name][src_key](), False)
                    action.add_effect(self.fluent['at'][name][dst_key](), True)
                    action.add_effect(self.fluent['free'][src_key](), True)
                    action.add_effect(self.fluent['free'][dst_key](), False)
                    if direction == 'up':
                        action.add_increase_effect(self.fluent['agent_y'][name](), 1)
                    elif direction == 'down':
                        action.add_decrease_effect(self.fluent['agent_y'][name](), 1)
                    elif direction == 'left':
                        action.add_decrease_effect(self.fluent['agent_x'][name](), 1)
                    else:
                        action.add_increase_effect(self.fluent['agent_x'][name](), 1)
                    agent.add_action(action)
                    if getattr(self.problem, 'waitfor', None) is not None:
                        self.problem.waitfor.annotate_as_waitfor(name, action.name, self.fluent['free'][dst_key]())

    def generate_problem(self, file_name, sl=False):
        self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('grid_' + file_name.replace('.json', ''))
        self.load_agents()
        self._create_cell_fluents()
        allowed_directions = self._allowed_directions()

        for agent in self.problem.agents:
            self._create_agent_fluents(agent)

        for agent in self.problem.agents:
            self._add_appear_action(agent)
            self._add_move_actions(agent, sl, allowed_directions)
            self._add_leave_action(agent)
            agent.add_public_goal(self.fluent['left'][agent.name]())

        for agent_name in self.instance_data['agents']:
            self._set_agent_initial_values(agent_name)

        return self.problem
