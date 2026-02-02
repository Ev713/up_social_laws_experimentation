import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


class ZenoTravelGenerator(ProblemGenerator):
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
        return zenotravel_sl.compile(self.problem).problem

    def generate_problem(self, file_name, sl=False):
        self.problem = MultiAgentProblemWithWaitfor('Zenotravel')
        self.load_instance_data(file_name)

        # Object types
        city = UserType('city')
        flevel = UserType('flevel')
        person = UserType('person')

        self.remember_obj_types(['citys', 'flevels', 'persons'], [city, flevel, person])

        citys = list(map(lambda c: unified_planning.model.Object(c, city), self.instance_data['citys']))
        self.problem.add_objects(citys)
        flevels = list(map(lambda f: unified_planning.model.Object(f, flevel), self.instance_data['flevels']))
        self.problem.add_objects(flevels)
        persons = list(map(lambda p: unified_planning.model.Object(p, person), self.instance_data['persons']))
        self.problem.add_objects(persons)

        # Public fluents
        person_at = Fluent('person_at', BoolType(), x=person, c=city)
        next = Fluent('next', BoolType(), l1=flevel, l2=flevel)
        self.problem.ma_environment.add_fluent(person_at, default_initial_value=False)
        self.problem.ma_environment.add_fluent(next, default_initial_value=False)

        # Agent fluents
        fuel_level = Fluent('fuel-level', BoolType(), l=flevel)
        carries = Fluent('carries', BoolType(), p=person)
        aircraft_at = Fluent('aircraft_at', BoolType(), c=city)

        # Actions
        board = InstantaneousAction('board', p=person, c=city)
        p = board.parameter('p')
        c = board.parameter('c')
        board.add_precondition(person_at(p, c))
        board.add_precondition(aircraft_at(c))
        board.add_effect(carries(p), True)
        board.add_effect(person_at(p, c), False)

        debark = InstantaneousAction('debark', p=person, c=city)
        p = debark.parameter('p')
        c = debark.parameter('c')
        debark.add_precondition(carries(p))
        debark.add_precondition(aircraft_at(c))
        debark.add_effect(person_at(p, c), True)
        debark.add_effect(carries(p), False)

        fly = InstantaneousAction('fly', c1=city, c2=city, l1=flevel, l2=flevel)
        c1 = fly.parameter('c1')
        c2 = fly.parameter('c2')
        l1 = fly.parameter('l1')
        l2 = fly.parameter('l2')
        fly.add_precondition(aircraft_at(c1))
        fly.add_precondition(fuel_level(l1))
        fly.add_precondition(next(l2, l1))
        fly.add_effect(aircraft_at(c2), True)
        fly.add_effect(fuel_level(l2), True)
        fly.add_effect(aircraft_at(c1), False)
        fly.add_effect(fuel_level(l1), False)

        zoom = InstantaneousAction('zoom', c1=city, c2=city, l1=flevel, l2=flevel, l3=flevel)
        c1 = zoom.parameter('c1')
        c2 = zoom.parameter('c2')
        l1 = zoom.parameter('l1')
        l2 = zoom.parameter('l2')
        l3 = zoom.parameter('l3')
        zoom.add_precondition(aircraft_at(c1))
        zoom.add_precondition(fuel_level(l1))
        zoom.add_precondition(next(l2, l1))
        zoom.add_precondition(next(l3, l2))
        zoom.add_effect(aircraft_at(c2), True)
        zoom.add_effect(fuel_level(l3), True)
        zoom.add_effect(aircraft_at(c1), False)
        zoom.add_effect(fuel_level(l1), True)

        refuel = InstantaneousAction('refuel', c=city, l=flevel, l1=flevel)
        c = refuel.parameter('c')
        l = refuel.parameter('l')
        l1 = refuel.parameter('l1')
        refuel.add_precondition(fuel_level(l))
        refuel.add_precondition(next(l, l1))
        refuel.add_precondition(aircraft_at(c))
        refuel.add_effect(fuel_level(l1), True)
        refuel.add_effect(fuel_level(l), False)

        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            agent.add_fluent(fuel_level, default_initial_value=False)
            agent.add_fluent(carries, default_initial_value=False)
            agent.add_fluent(aircraft_at, default_initial_value=False)
            agent.add_action(board)
            agent.add_action(debark)
            agent.add_action(fly)
            agent.add_action(zoom)
            agent.add_action(refuel)
            self.problem.add_agent(agent)

        self.set_init_values()
        self.set_goals()

        if sl:
            self.add_social_law()

        return self.problem
