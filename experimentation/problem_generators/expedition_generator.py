import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Equals, And, Not, IntType, GE, LE, Minus, Plus, GT

from experimentation.problem_generators.problem_generator import ProblemGenerator, NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


class ExpeditionGenerator(NumericProblemGenerator):

    def add_social_law(self):
        sl = SocialLaw()
        sl.skip_checks = True
        starting_loc = {}
        for a in self.problem.agents:
            # print(a.name)
            starting_loc[a.name] = None
            for w in self.instance_data['waypoint']:
                init_values = self.instance_data['init_values'][a.name]
                for init_val in init_values:
                    if init_val[0] == 'at' and len(init_val) > 0 and w in init_val[1]:
                        starting_loc[a.name] = w
                        break
                    if starting_loc[a.name] is not None:
                        break
            if starting_loc[a.name] is None:
                raise Exception(f'Agent {a.name} doesn\'t have a starting location')
            # print(f'starting loc: {starting_loc[a.name]}')
        for a in self.problem.agents:
            packs = None
            for init_val in self.instance_data['init_values']['global']:
                if init_val[0] == '=' and init_val[1][1][0] == starting_loc[a.name]:
                    packs = int(int(init_val[2])/len([x for x in starting_loc if starting_loc[x] == starting_loc[a.name]]))
                    # print(f'{a.name} packs: {packs}')
                    break
            if packs is None:
                raise Exception(f'Can\'t find Agent {a.name}\' packs!')
            sl.add_new_fluent(a.name, 'personal_packs', (('w', 'waypoint'), ), 0)
            sl.add_precondition_to_action(a.name, 'retrieve_supplies', 'personal_packs', ('w',), '>=', 1)
            sl.add_effect(a.name, 'retrieve_supplies', 'personal_packs', ('w',), 1, '-')
            sl.add_effect(a.name, 'store_supplies', 'personal_packs', ('w',), 1, '+')
            sl.set_initial_value_for_new_fluent(a.name, 'personal_packs', (starting_loc[a.name],), packs)
        self.problem = sl.compile(self.problem).problem
        return self.problem


    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('expedition_' + file_name.replace('.json', ''))
        self.load_instance_data(file_name)

        waypoint = UserType('waypoint')
        at = Fluent('at', BoolType(), w=waypoint)
        is_next = Fluent('is_next', BoolType(), x=waypoint, y=waypoint)
        sled_supplies = Fluent('sled_supplies', IntType())
        sled_capacity = Fluent('sled_capacity', IntType())
        waypoint_supplies = Fluent('waypoint_supplies', IntType(), w=waypoint)

        self.load_objects(['waypoint'], [waypoint])

        self.problem.ma_environment.add_fluent(is_next, default_initial_value=False)
        self.problem.ma_environment.add_fluent(waypoint_supplies, default_initial_value=0)

        move = {}
        for dir in ['forwards', 'backwards']:
            move[dir] = InstantaneousAction(f'move_{dir}', w1=waypoint, w2=waypoint)
            w1 = move[dir].parameter('w1')
            w2 = move[dir].parameter('w2')
            move[dir].add_precondition(at(w1))
            if dir == 'forwards':
                move[dir].add_precondition(is_next(w1, w2))
            else:
                move[dir].add_precondition(is_next(w2, w1))
            move[dir].add_precondition(GE(sled_supplies, 1))
            move[dir].add_effect(at(w1), False)
            move[dir].add_effect(at(w2), True)
            move[dir].add_effect(sled_supplies, Minus(sled_supplies, 1))

        store_supplies = InstantaneousAction('store_supplies', w=waypoint)
        w = store_supplies.parameter('w')
        store_supplies.add_precondition(at(w))
        store_supplies.add_precondition(GE(sled_supplies, 1))
        store_supplies.add_effect(sled_supplies, Minus(sled_supplies, 1))
        store_supplies.add_effect(waypoint_supplies(w), Plus(waypoint_supplies(w), 1))

        retrieve_supplies = InstantaneousAction('retrieve_supplies', w=waypoint)
        w = retrieve_supplies.parameter('w')
        retrieve_supplies.add_precondition(at(w))
        retrieve_supplies.add_precondition(GE(waypoint_supplies(w), 1))
        retrieve_supplies.add_precondition(GT(sled_capacity, sled_supplies))
        retrieve_supplies.add_effect(sled_supplies, Plus(sled_supplies, 1))
        retrieve_supplies.add_effect(waypoint_supplies(w), Minus(waypoint_supplies(w), 1))

        self.load_agents()
        for a in self.problem.agents:
            a.add_action(move['forwards'])
            a.add_action(move['backwards'])
            a.add_action(store_supplies)
            a.add_action(retrieve_supplies)
            a.add_fluent(sled_supplies, default_initial_value=0)
            a.add_fluent(sled_capacity, default_initial_value=0)
            a.add_fluent(at, default_initial_value=False)

        self.set_init_values()
        self.set_goals()
        if sl:
            self.add_social_law()
        return self.problem
