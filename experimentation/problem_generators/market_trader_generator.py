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
    def _get_market_trader_bounds(self):
        bounds = self.instance_data.get('bounds', {})
        max_units = 0
        total_units = 0
        max_initial_capacity = 0

        for fluentuple in self.instance_data.get('init_values', {}).get('global', []):
            if fluentuple[0] == '=' and fluentuple[1][0] == 'on-sale':
                units = int(float(fluentuple[2]))
                max_units = max(max_units, units)
                total_units += units

        for agent_name in self.instance_data.get('agents', []):
            for fluentuple in self.instance_data.get('init_values', {}).get(agent_name, []):
                if fluentuple[0] == '=' and fluentuple[1][0] == 'capacity':
                    max_initial_capacity = max(max_initial_capacity, int(float(fluentuple[2])))

        inventory_bound = bounds.get('inventory', bounds.get('on-sale', max_units))
        capacity_bound = bounds.get('capacity', max_initial_capacity + total_units + 20)

        if inventory_bound < 0 or capacity_bound < 0:
            raise ValueError("Market Trader bounds must be non-negative.")

        return {
            'inventory': inventory_bound,
            'capacity': capacity_bound,
        }

    def _get_numeric_init_value(self, scope, fluent_name, args, default=None):
        for fluentuple in self.instance_data.get('init_values', {}).get(scope, []):
            if fluentuple[0] != '=':
                continue
            target = fluentuple[1]
            if target[0] == fluent_name and tuple(target[1]) == tuple(args):
                value = float(fluentuple[2])
                if value % 1 == 0:
                    return int(value)
                return value
        if default is not None:
            return default
        raise ValueError(f"Missing initial numeric value for {scope}:{fluent_name}{tuple(args)}")

    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('market_trader_' + file_name.replace('.json', ''))
        self.load_instance_data(file_name)
        bounds = self._get_market_trader_bounds()
        market = UserType('market')
        goods = UserType('goods')
        self.load_objects(['market', 'goods'], [market, goods])
        on_sale = Fluent('on-sale', IntType(0, bounds['inventory']), g=goods, m=market)
        drive_cost = Fluent('drive-cost', RealType(), m1=market, m2=market)
        price = Fluent('price', RealType(), g=goods, m=market)
        sellprice = Fluent('sellprice', RealType(), g=goods, m=market)
        self.problem.ma_environment.add_fluent(on_sale, default_initial_value=0)
        self.problem.ma_environment.add_fluent(drive_cost, default_initial_value=0)
        self.problem.ma_environment.add_fluent(price, default_initial_value=0)
        self.problem.ma_environment.add_fluent(sellprice, default_initial_value=0)

        bought = Fluent('bought', IntType(0, bounds['inventory']), g=goods)
        cash = Fluent('cash', RealType())
        capacity = Fluent('capacity', IntType(0, bounds['capacity']))
        at = Fluent('at', BoolType(), m=market)
        can_drive = Fluent('can-drive', BoolType(), m1=market, m2=market)

        def inc(action, fluent_exp, amount):
            action.add_increase_effect(fluent_exp, amount)

        def dec(action, fluent_exp, amount):
            action.add_decrease_effect(fluent_exp, amount)

        upgrade = InstantaneousAction('upgrade', )
        upgrade.add_precondition(GE(cash, 50))
        dec(upgrade, cash, 50)
        inc(upgrade, capacity, 20)
        self.load_agents()
        for a in self.problem.agents:
            a.add_fluent(at, default_initial_value=False)
            a.add_fluent(can_drive, default_initial_value=False)
            a.add_fluent(bought, default_initial_value=0)
            a.add_fluent(cash, default_initial_value=0)
            a.add_fluent(capacity, default_initial_value=0)
            a.add_action(upgrade)

            for m1_name in self.instance_data['market']:
                for m2_name in self.instance_data['market']:
                    if m1_name == m2_name:
                        continue
                    travel_cost = self._get_numeric_init_value('global', 'drive-cost', [m1_name, m2_name], default=0)
                    travel = InstantaneousAction(f'travel_{m1_name}_{m2_name}')
                    travel.add_precondition(can_drive(
                        unified_planning.model.Object(m1_name, market),
                        unified_planning.model.Object(m2_name, market),
                    ))
                    travel.add_precondition(GE(cash, travel_cost))
                    travel.add_precondition(at(unified_planning.model.Object(m1_name, market)))
                    dec(travel, cash, travel_cost)
                    travel.add_effect(at(unified_planning.model.Object(m1_name, market)), False)
                    travel.add_effect(at(unified_planning.model.Object(m2_name, market)), True)
                    a.add_action(travel)

            for goods_name in self.instance_data['goods']:
                goods_obj = unified_planning.model.Object(goods_name, goods)
                for market_name in self.instance_data['market']:
                    market_obj = unified_planning.model.Object(market_name, market)
                    buy_price = self._get_numeric_init_value('global', 'price', [goods_name, market_name], default=0)
                    sell_price = self._get_numeric_init_value('global', 'sellprice', [goods_name, market_name], default=0)

                    buy = InstantaneousAction(f'buy_{goods_name}_{market_name}')
                    buy.add_precondition(at(market_obj))
                    buy.add_precondition(LE(buy_price, cash))
                    buy.add_precondition(GE(capacity, 1))
                    buy.add_precondition(GE(on_sale(goods_obj, market_obj), 1))
                    dec(buy, capacity, 1)
                    dec(buy, on_sale(goods_obj, market_obj), 1)
                    inc(buy, bought(goods_obj), 1)
                    dec(buy, cash, buy_price)
                    a.add_action(buy)

                    sell = InstantaneousAction(f'sell_{goods_name}_{market_name}')
                    sell.add_precondition(at(market_obj))
                    sell.add_precondition(GE(bought(goods_obj), 1))
                    inc(sell, capacity, 1)
                    dec(sell, bought(goods_obj), 1)
                    inc(sell, cash, sell_price)
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
