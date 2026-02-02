import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType, Equals, And, Not, IntType, GE, LE, Minus, Plus, GT, RealType, \
    LT, Times

from experimentation.problem_generators.problem_generator import ProblemGenerator, NumericProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw

class NumericZenotravelGenerator(NumericProblemGenerator):
    def __init__(self):
        super().__init__()

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

        # Object types
        city = UserType('city')
        person = UserType('person')

        self.remember_obj_types(['city', 'person'], [city, person])
        self.load_objects(['city', 'person'], [city, person])

        # Public fluents
        person_loc = Fluent('person-loc', BoolType(), x=person, c=city)
        self.problem.ma_environment.add_fluent(person_loc, default_initial_value=False)
        distance = Fluent('distance', RealType(), c1=city, c2=city)
        self.problem.ma_environment.add_fluent(distance, default_initial_value=False)

        # Agent fluents
        carries = Fluent('carries', BoolType(), p=person)
        fuel = Fluent('fuel', RealType(), )
        slow_burn = Fluent('slow-burn', RealType(), )
        fast_burn = Fluent('fast-burn', RealType(), )
        capacity = Fluent('capacity', RealType(), )
        onboard = Fluent('onboard', RealType(), )
        zoom_limit = Fluent('zoom-limit', RealType(), )
        aircraft_loc = Fluent('aircraft-loc', BoolType(), c=city)

        # Actions
        board = InstantaneousAction('board', p=person, c=city)
        p = board.parameter('p')
        c = board.parameter('c')
        board.add_precondition(person_loc(p, c))
        board.add_precondition(aircraft_loc(c))
        board.add_effect(onboard, Plus(onboard + 1))
        board.add_effect(carries(p), True)
        board.add_effect(person_loc(p, c), False)

        debark = InstantaneousAction('debark', p=person, c=city)
        p = debark.parameter('p')
        c = debark.parameter('c')
        debark.add_precondition(carries(p))
        debark.add_precondition(aircraft_loc(c))
        debark.add_effect(onboard, Minus(onboard, 1))
        debark.add_effect(person_loc(p, c), True)
        debark.add_effect(carries(p), False)

        fly_slow = InstantaneousAction('fly-slow', c1=city, c2=city)
        c1 = fly_slow.parameter('c1')
        c2 = fly_slow.parameter('c2')
        fly_slow.add_precondition(aircraft_loc(c1))
        fly_slow.add_precondition(GE(fuel, Times(distance(c1, c2), slow_burn)))
        fly_slow.add_precondition(GT(distance(c1, c2), 0))

        fly_slow.add_effect(aircraft_loc(c2), True)
        fly_slow.add_effect(aircraft_loc(c1), False)
        fly_slow.add_effect(fuel, Minus(fuel, Times(distance(c1, c2), slow_burn)))

        fly_fast = InstantaneousAction('fly-fast', c1=city, c2=city, )
        c1 = fly_fast.parameter('c1')
        c2 = fly_fast.parameter('c2')
        fly_fast.add_precondition(aircraft_loc(c1))
        fly_fast.add_precondition(GT(distance(c1, c2), 0))
        fly_fast.add_precondition(GE(fuel, Times(distance(c1, c2), fast_burn)))
        fly_fast.add_precondition(GE(zoom_limit, onboard))
        fly_fast.add_effect(aircraft_loc(c2), True)
        fly_fast.add_effect(aircraft_loc(c1), False)
        fly_fast.add_effect(fuel, Minus(fuel, Times(distance(c1, c2), fast_burn)))

        refuel = InstantaneousAction('refuel', )
        refuel.add_precondition(GT(capacity, fuel))
        refuel.add_effect(fuel, capacity)

        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            agent.add_fluent(fuel, default_initial_value=0)
            agent.add_fluent(carries, default_initial_value=False)
            agent.add_fluent(aircraft_loc, default_initial_value=False)
            agent.add_fluent(capacity, default_initial_value=0)
            agent.add_fluent(fast_burn, default_initial_value=0)
            agent.add_fluent(slow_burn, default_initial_value=0)
            agent.add_fluent(onboard, default_initial_value=0)
            agent.add_fluent(zoom_limit, default_initial_value=0)

            agent.add_action(board)
            agent.add_action(debark)
            agent.add_action(fly_slow)
            agent.add_action(fly_fast)
            agent.add_action(refuel)
            self.problem.add_agent(agent)

        self.set_init_values()
        self.set_goals()

        if sl:
            self.add_social_law()
        return self.problem