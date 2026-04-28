import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, IntType, GE

from experimentation.problem_generators.problem_generator import NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw

class NumericZenotravelGenerator(NumericProblemGenerator):
    def __init__(self):
        super().__init__()

    def _get_numeric_init_value(self, scope, fluent_name, args=(), default=None):
        for fluentuple in self.instance_data.get('init_values', {}).get(scope, []):
            if fluentuple[0] != '=':
                continue
            target = fluentuple[1]
            if target[0] == fluent_name and tuple(target[1]) == tuple(args):
                value = int(float(fluentuple[2]))
                return value
        if default is not None:
            return default
        raise ValueError(f"Missing initial numeric value for {scope}:{fluent_name}{tuple(args)}")

    def _get_bounds(self):
        bounds = self.instance_data.get('bounds', {})
        people_count = len(self.instance_data['person'])
        max_distance = 0
        max_capacity = 0
        max_fuel = 0
        max_slow_burn = 0
        max_fast_burn = 0
        max_zoom_limit = 0

        for fluentuple in self.instance_data.get('init_values', {}).get('global', []):
            if fluentuple[0] == '=' and fluentuple[1][0] == 'distance':
                max_distance = max(max_distance, int(float(fluentuple[2])))

        for agent_name in self.instance_data.get('agents', []):
            max_capacity = max(max_capacity, self._get_numeric_init_value(agent_name, 'capacity'))
            max_fuel = max(max_fuel, self._get_numeric_init_value(agent_name, 'fuel'))
            max_slow_burn = max(max_slow_burn, self._get_numeric_init_value(agent_name, 'slow-burn'))
            max_fast_burn = max(max_fast_burn, self._get_numeric_init_value(agent_name, 'fast-burn'))
            max_zoom_limit = max(max_zoom_limit, self._get_numeric_init_value(agent_name, 'zoom-limit'))

        return {
            'distance': bounds.get('distance', max_distance),
            'fuel': bounds.get('fuel', max_capacity),
            'fuel-space': bounds.get('fuel-space', max_capacity),
            'capacity': bounds.get('capacity', max_capacity),
            'slow-burn': bounds.get('slow-burn', max_slow_burn),
            'fast-burn': bounds.get('fast-burn', max_fast_burn),
            'onboard': bounds.get('onboard', people_count),
            'zoom-limit': bounds.get('zoom-limit', max_zoom_limit),
            'fast-room-low': bounds.get('fast-room-low', -people_count),
            'fast-room-high': bounds.get('fast-room-high', max_zoom_limit),
        }

    def _get_refuel_amounts(self, capacity):
        amounts = []
        amount = 1
        while amount <= capacity:
            amounts.append(amount)
            amount *= 2
        return amounts

    def add_social_law(self):
        zenotravel_sl = SocialLaw()
        for agent in self.problem.agents:
            zenotravel_sl.add_new_fluent(agent.name, 'assigned', (('p', 'person'),), False)
        persons_to_aircraft = {}
        for agent in self.problem.agents:
            for goal in agent.public_goals:
                args = [arg.object() for arg in goal.args if arg.is_object_exp()]
                persons_args = [obj for obj in args if obj.type.name == 'person']
                for person in persons_args:
                    persons_to_aircraft[person.name] = agent.name
        for agent in self.problem.agents:
            zenotravel_sl.add_precondition_to_action(agent.name, 'board', 'assigned', ('p',))
        for person in self.problem.objects(UserType('person')):
            if person.name in persons_to_aircraft:
                aircraft_name = persons_to_aircraft[person.name]
                zenotravel_sl.set_initial_value_for_new_fluent(aircraft_name, 'assigned', (person.name,), True)
        zenotravel_sl.skip_checks = True
        self.problem = zenotravel_sl.compile(self.problem).problem
        return self.problem

    def generate_problem(self, file_name, sl=False):
        self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('zenotravel_' + file_name.replace('.json', ''))
        self.agent_type_name = 'plane'
        bounds = self._get_bounds()

        # Object types
        city = UserType('city')
        person = UserType('person')

        self.remember_obj_types(['city', 'person'], [city, person])
        self.load_objects(['city', 'person'], [city, person])

        # Public fluents
        person_loc = Fluent('person-loc', BoolType(), x=person, c=city)
        self.problem.ma_environment.add_fluent(person_loc, default_initial_value=False)
        distance = Fluent('distance', IntType(0, bounds['distance']), c1=city, c2=city)
        self.problem.ma_environment.add_fluent(distance, default_initial_value=0)

        # Agent fluents
        carries = Fluent('carries', BoolType(), p=person)
        fuel = Fluent('fuel', IntType(0, bounds['fuel']))
        fuel_space = Fluent('fuel-space', IntType(0, bounds['fuel-space']))
        slow_burn = Fluent('slow-burn', IntType(0, bounds['slow-burn']))
        fast_burn = Fluent('fast-burn', IntType(0, bounds['fast-burn']))
        capacity = Fluent('capacity', IntType(0, bounds['capacity']))
        onboard = Fluent('onboard', IntType(0, bounds['onboard']))
        zoom_limit = Fluent('zoom-limit', IntType(0, bounds['zoom-limit']))
        fast_room = Fluent('fast-room', IntType(bounds['fast-room-low'], bounds['fast-room-high']))
        aircraft_loc = Fluent('aircraft-loc', BoolType(), c=city)

        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            agent_capacity = self._get_numeric_init_value(agent_name, 'capacity')
            agent_fuel = self._get_numeric_init_value(agent_name, 'fuel')
            agent_slow_burn = self._get_numeric_init_value(agent_name, 'slow-burn')
            agent_fast_burn = self._get_numeric_init_value(agent_name, 'fast-burn')
            agent_zoom_limit = self._get_numeric_init_value(agent_name, 'zoom-limit')
            agent_onboard = self._get_numeric_init_value(agent_name, 'onboard', default=0)

            agent.add_fluent(fuel, default_initial_value=0)
            agent.add_fluent(fuel_space, default_initial_value=0)
            agent.add_fluent(carries, default_initial_value=False)
            agent.add_fluent(aircraft_loc, default_initial_value=False)
            agent.add_fluent(capacity, default_initial_value=0)
            agent.add_fluent(fast_burn, default_initial_value=0)
            agent.add_fluent(slow_burn, default_initial_value=0)
            agent.add_fluent(onboard, default_initial_value=0)
            agent.add_fluent(zoom_limit, default_initial_value=0)
            agent.add_fluent(fast_room, default_initial_value=0)

            board = InstantaneousAction('board', p=person, c=city)
            p = board.parameter('p')
            c = board.parameter('c')
            board.add_precondition(person_loc(p, c))
            board.add_precondition(aircraft_loc(c))
            board.add_increase_effect(onboard, 1)
            board.add_decrease_effect(fast_room, 1)
            board.add_effect(carries(p), True)
            board.add_effect(person_loc(p, c), False)

            debark = InstantaneousAction('debark', p=person, c=city)
            p = debark.parameter('p')
            c = debark.parameter('c')
            debark.add_precondition(carries(p))
            debark.add_precondition(aircraft_loc(c))
            debark.add_decrease_effect(onboard, 1)
            debark.add_increase_effect(fast_room, 1)
            debark.add_effect(person_loc(p, c), True)
            debark.add_effect(carries(p), False)

            agent.add_action(board)
            agent.add_action(debark)

            for c1_name in self.instance_data['city']:
                for c2_name in self.instance_data['city']:
                    if c1_name == c2_name:
                        continue
                    dist = self._get_numeric_init_value('global', 'distance', [c1_name, c2_name], default=0)
                    if dist <= 0:
                        continue
                    c1_obj = unified_planning.model.Object(c1_name, city)
                    c2_obj = unified_planning.model.Object(c2_name, city)

                    slow_cost = dist * agent_slow_burn
                    if slow_cost <= agent_capacity:
                        fly_slow = InstantaneousAction(f'fly-slow_{c1_name}_{c2_name}')
                        fly_slow.add_precondition(aircraft_loc(c1_obj))
                        fly_slow.add_precondition(GE(fuel, slow_cost))
                        fly_slow.add_effect(aircraft_loc(c2_obj), True)
                        fly_slow.add_effect(aircraft_loc(c1_obj), False)
                        fly_slow.add_decrease_effect(fuel, slow_cost)
                        fly_slow.add_increase_effect(fuel_space, slow_cost)
                        agent.add_action(fly_slow)

                    fast_cost = dist * agent_fast_burn
                    if fast_cost <= agent_capacity:
                        fly_fast = InstantaneousAction(f'fly-fast_{c1_name}_{c2_name}')
                        fly_fast.add_precondition(aircraft_loc(c1_obj))
                        fly_fast.add_precondition(GE(fuel, fast_cost))
                        fly_fast.add_precondition(GE(fast_room, 0))
                        fly_fast.add_effect(aircraft_loc(c2_obj), True)
                        fly_fast.add_effect(aircraft_loc(c1_obj), False)
                        fly_fast.add_decrease_effect(fuel, fast_cost)
                        fly_fast.add_increase_effect(fuel_space, fast_cost)
                        agent.add_action(fly_fast)

            for amount in self._get_refuel_amounts(agent_capacity):
                refuel = InstantaneousAction(f'refuel_{amount}')
                refuel.add_precondition(GE(fuel_space, amount))
                refuel.add_increase_effect(fuel, amount)
                refuel.add_decrease_effect(fuel_space, amount)
                agent.add_action(refuel)

            self.problem.add_agent(agent)

        self.set_init_values()
        for agent_name in self.instance_data['agents']:
            agent = self.problem.agent(agent_name)
            agent_capacity = self._get_numeric_init_value(agent_name, 'capacity')
            agent_fuel = self._get_numeric_init_value(agent_name, 'fuel')
            agent_zoom_limit = self._get_numeric_init_value(agent_name, 'zoom-limit')
            agent_onboard = self._get_numeric_init_value(agent_name, 'onboard', default=0)
            self.problem.set_initial_value(unified_planning.shortcuts.Dot(agent, fuel_space), agent_capacity - agent_fuel)
            self.problem.set_initial_value(unified_planning.shortcuts.Dot(agent, fast_room), agent_zoom_limit - agent_onboard)
        self.set_goals()

        if sl:
            self.add_social_law()
        return self.problem
