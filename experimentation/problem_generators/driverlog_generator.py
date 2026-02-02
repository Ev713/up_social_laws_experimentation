import unified_planning
from unified_planning.model import Fluent, InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import UserType, BoolType

from experimentation.problem_generators.problem_generator import ProblemGenerator
from up_social_laws.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from up_social_laws.social_law import SocialLaw


class DriverLogGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()

    def add_social_law(self):
        driverlog_sl = SocialLaw()
        for agent in self.problem.agents:
            driverlog_sl.add_new_fluent(agent.name, 'trunk_empty', (("t", "truck"),), True)
            driverlog_sl.add_new_fluent(None, f'{agent.name}_can_board', (("t", "truck"),), True)
            driverlog_sl.add_effect(agent.name, 'LOAD-TRUCK', 'trunk_empty', ('truck',), False)
            driverlog_sl.add_effect(agent.name, 'UNLOAD-TRUCK', 'trunk_empty', ('truck',), True)
            driverlog_sl.add_waitfor_annotation(agent.name, 'BOARD-TRUCK', 'at', ('truck', 'loc'))
            driverlog_sl.add_waitfor_annotation(agent.name, 'BOARD-TRUCK', 'empty', ('truck',))

        for agent in self.problem.agents:
            for other_agent in self.problem.agents:
                if other_agent.name == agent.name:
                    continue
                driverlog_sl.add_effect(agent.name, 'BOARD-TRUCK', f'{other_agent.name}_can_board', ('truck',), False)
            driverlog_sl.add_precondition_to_action(agent.name, 'BOARD-TRUCK', f'{agent.name}_can_board', ('truck',))

        for truck in self.problem.objects(UserType('truck', father=UserType('locatable'))):
            driverlog_sl.add_public_goal('empty', (truck.name,))
            driverlog_sl.add_public_goal('empty', (truck.name,))

        # add precondition board driver's truck

        return driverlog_sl.compile(self.problem).problem

    def generate_problem(self, file_name, sl=False):
        self.instance_data = self.load_instance_data(file_name)
        self.problem = MultiAgentProblemWithWaitfor('driverlog')

        # Objects
        locatable = UserType('locatable')
        location = UserType('location')
        truck = UserType('truck', father=locatable)
        package = UserType('package', father=locatable)

        self.remember_obj_types(['trucks', 'packages', 'locations'], [truck, package, location])

        # General fluents
        in_ = Fluent('in', BoolType(), obj1=package, obj=truck)
        path = Fluent('path', BoolType(), x=location, y=location)
        empty = Fluent('empty', BoolType(), v=truck)
        at = Fluent('at', BoolType(), obj=locatable, loc=location)
        link = Fluent('link', BoolType(), x=location, y=location)

        self.problem.ma_environment.add_fluent(in_, default_initial_value=False)
        self.problem.ma_environment.add_fluent(path, default_initial_value=False)
        self.problem.ma_environment.add_fluent(empty, default_initial_value=False)
        self.problem.ma_environment.add_fluent(at, default_initial_value=False)
        self.problem.ma_environment.add_fluent(link, default_initial_value=False)

        # Objects
        locations = list(map(lambda l: unified_planning.model.Object(l, location), self.instance_data['locations']))
        self.problem.add_objects(locations)
        trucks = list(map(lambda t: unified_planning.model.Object(t, truck), self.instance_data['trucks']))
        self.problem.add_objects(trucks)
        packages = list(map(lambda p: unified_planning.model.Object(p, package), self.instance_data['packages']))
        self.problem.add_objects(packages)

        # Agent specific fluents
        driving = Fluent('driving', BoolType(), v=truck)
        driver_at = Fluent('driver_at', BoolType(), loc=location)

        # Actions
        load_truck = InstantaneousAction('LOAD-TRUCK', truck=truck, obj=package, loc=location)
        tr = load_truck.parameter('truck')
        o = load_truck.parameter('obj')
        l = load_truck.parameter('loc')
        load_truck.add_precondition(at(tr, l))
        load_truck.add_precondition(at(o, l))
        load_truck.add_precondition(driving(tr))
        load_truck.add_effect(at(o, l), False)
        load_truck.add_effect(in_(o, tr), True)

        unload_truck = InstantaneousAction('UNLOAD-TRUCK', truck=truck, obj=package, loc=location)
        tr = unload_truck.parameter('truck')
        o = unload_truck.parameter('obj')
        l = unload_truck.parameter('loc')
        unload_truck.add_precondition(at(tr, l))
        unload_truck.add_precondition(in_(o, tr))
        unload_truck.add_precondition(driving(tr))
        unload_truck.add_effect(at(o, l), True)
        unload_truck.add_effect(in_(o, tr), False)

        board_truck = InstantaneousAction('BOARD-TRUCK', loc=location, truck=truck)
        l = board_truck.parameter('loc')
        tr = board_truck.parameter('truck')
        board_truck.add_precondition(at(tr, l))
        board_truck.add_precondition(driver_at(l))
        board_truck.add_precondition(empty(tr))
        board_truck.add_effect(driver_at(l), False)
        board_truck.add_effect(driving(tr), True)
        board_truck.add_effect(empty(tr), False)

        disembark_truck = InstantaneousAction('DISEMBARK-TRUCK', loc=location, truck=truck)
        l = disembark_truck.parameter('loc')
        tr = disembark_truck.parameter('truck')
        disembark_truck.add_precondition(at(tr, l))
        disembark_truck.add_precondition(driving(tr))
        disembark_truck.add_effect(driving(tr), False)
        disembark_truck.add_effect(driver_at(l), True)
        disembark_truck.add_effect(empty(tr), True)

        drive_truck = InstantaneousAction('DRIVE-TRUCK', from_=location, to=location, truck=truck)
        from_ = drive_truck.parameter('from_')
        to = drive_truck.parameter('to')
        tr = drive_truck.parameter('truck')
        drive_truck.add_precondition(at(tr, from_))
        drive_truck.add_precondition(driving(tr))
        drive_truck.add_precondition(link(from_, to))
        drive_truck.add_effect(at(tr, to), True)
        drive_truck.add_effect(at(tr, from_), False)

        walk = InstantaneousAction('WALK', from_=location, to=location)
        from_ = walk.parameter('from_')
        to = walk.parameter('to')
        walk.add_precondition(driver_at(from_))
        walk.add_precondition(path(from_, to))
        walk.add_effect(driver_at(to), True)
        walk.add_effect(driver_at(from_), False)

        # Agents
        for agent_name in self.instance_data['agents']:
            agent = Agent(agent_name, self.problem)
            self.problem.add_agent(agent)
            agent.add_fluent(driver_at, default_initial_value=False)
            agent.add_fluent(driving, default_initial_value=False)
            agent.add_action(load_truck)
            agent.add_action(unload_truck)
            agent.add_action(board_truck)
            agent.add_action(disembark_truck)
            agent.add_action(drive_truck)
            agent.add_action(walk)

        self.set_init_values()
        self.set_goals()
        if sl:
            self.add_social_law()
        return self.problem