import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Equals, And, Not, IntType, GE, LE, Minus, Plus, GT, RealType, \
    LT

from experimentation.problem_generators.problem_generator import ProblemGenerator, NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw

OPERATORS = {
    '=': Equals,
    '>=': GE,
    '>': GT,
    '<=': LE,
    '<': LT
}


class MarketTraderGenerator(NumericProblemGenerator):
    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('market_trader_' + file_name.replace('.json', ''))
        self.load_instance_data(file_name)
        market = UserType('market')
        goods = UserType('goods')
        self.load_objects(['market', 'goods'], [market, goods])
        on_sale = Fluent('on-sale', IntType(), g=goods, m=market)
        drive_cost = Fluent('drive-cost', RealType(), m1=market, m2=market)
        price = Fluent('price', RealType(), g=goods, m=market)
        sellprice = Fluent('sellprice', RealType(), g=goods, m=market)
        self.problem.ma_environment.add_fluent(on_sale, default_initial_value=0)
        self.problem.ma_environment.add_fluent(drive_cost, default_initial_value=0)
        self.problem.ma_environment.add_fluent(price, default_initial_value=0)
        self.problem.ma_environment.add_fluent(sellprice, default_initial_value=0)

        bought = Fluent('bought', IntType(), g=goods)
        cash = Fluent('cash', RealType())
        capacity = Fluent('capacity', IntType())
        at = Fluent('at', BoolType(), m=market)
        can_drive = Fluent('can-drive', BoolType(), m1=market, m2=market)

        travel = InstantaneousAction('travel', m1=market, m2=market)
        m1 = travel.parameter('m1')
        m2 = travel.parameter('m2')
        travel.add_precondition(can_drive(m1, m2))
        travel.add_precondition(GE(cash, drive_cost(m1, m2)))
        travel.add_precondition(at(m1))
        travel.add_effect(cash, Minus(cash, drive_cost(m1, m2)))
        travel.add_effect(at(m1), False)
        travel.add_effect(at(m2), True)

        buy = InstantaneousAction('buy', g=goods, m=market)
        g = buy.parameter('g')
        m = buy.parameter('m')
        buy.add_precondition(at(m))
        buy.add_precondition(LE(price(g, m), cash))
        buy.add_precondition(GE(capacity, 1))
        buy.add_precondition(GT(on_sale(g, m), 0))
        buy.add_effect(capacity, Minus(capacity, 1))
        buy.add_effect(on_sale(g, m), Minus(on_sale(g, m), 1))
        buy.add_effect(bought(g), Plus(bought(g), 1))
        buy.add_effect(cash, Minus(cash, price(g, m)))

        upgrade = InstantaneousAction('upgrade', )
        upgrade.add_precondition(GE(cash, 5))
        upgrade.add_effect(cash, Minus(cash, 50))
        upgrade.add_effect(capacity, Plus(capacity, 20))

        sell = InstantaneousAction('sell', g=goods, m=market)
        sell.add_precondition(at(m))
        sell.add_precondition(GE(bought(g), 1))
        sell.add_effect(capacity, Plus(capacity, 1))
        sell.add_effect(bought(g), Minus(bought(g), 1))
        sell.add_effect(on_sale(g, m), Plus(on_sale(g, m), 1))
        sell.add_effect(cash, Plus(cash, sellprice(g, m)))
        self.load_agents()
        for a in self.problem.agents:
            a.add_fluent(at, default_initial_value=False)
            a.add_fluent(can_drive, default_initial_value=False)
            a.add_fluent(bought, default_initial_value=0)
            a.add_fluent(cash, default_initial_value=0)
            a.add_fluent(capacity, default_initial_value=0)
            a.add_action(travel)
            a.add_action(buy)
            a.add_action(upgrade)
            a.add_action(sell)

        self.set_init_values()
        self.set_goals()
        if sl:
            self.add_social_law()
        return self.problem

    def set_goals(self):
        assigned_index = 0
        num_of_agents = len(self.problem.agents)
        for agent_name in ['global'] + [agent.name for agent in self.problem.agents]:
            if agent_name not in self.instance_data['goals']:
                continue
            for goaltuple in self.instance_data['goals'][agent_name]:
                if agent_name == 'global':
                    agent = self.problem.agents[assigned_index]
                    assigned_index = (assigned_index + 1) % num_of_agents
                else:
                    agent = self.problem.agent(agent_name)
                if goaltuple[0] in OPERATORS:
                    expr = OPERATORS[goaltuple[0]] \
                        (*[self.create_fluent_expression(goal_expr,
                                                         None if agent_name == 'global' else agent) for goal_expr in
                           goaltuple[1:]])
                    agent.add_public_goal(expr)

                else:
                    agent.add_public_goal(
                        self.create_fluent_expression(goaltuple, None if agent_name == 'global' else agent))
