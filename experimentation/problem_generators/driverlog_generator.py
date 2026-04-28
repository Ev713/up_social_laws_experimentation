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

    def _agent_goal_map(self):
        goals_data = self.instance_data['goals']
        agent_names = self.instance_data['agents']

        if isinstance(goals_data, dict):
            return {agent_name: goals_data.get(agent_name, []) for agent_name in agent_names}

        agent_goals = {agent_name: [] for agent_name in agent_names}
        for goal_index, goaltuple in enumerate(goals_data):
            agent_name = agent_names[goal_index % len(agent_names)]
            agent_goals[agent_name].append(goaltuple)
        return agent_goals

    def add_social_law(self):
        driverlog_sl = SocialLaw()
        agent_goals = self._agent_goal_map()

        for agent_name, truck_name in zip(self.instance_data['agents'], self.instance_data['trucks']):
            driverlog_sl.add_new_fluent(agent_name, 'assigned_truck', (("t", "truck"),), False)
            driverlog_sl.set_initial_value_for_new_fluent(agent_name, 'assigned_truck', (truck_name,), True)
            driverlog_sl.add_new_fluent(agent_name, 'assigned_package', (("p", "package"),), False)

            for action_name in ['BOARD-TRUCK', 'DISEMBARK-TRUCK', 'DRIVE-TRUCK']:
                driverlog_sl.add_precondition_to_action(agent_name, action_name, 'assigned_truck', ('truck',))

            for action_name in ['LOAD-TRUCK', 'UNLOAD-TRUCK']:
                driverlog_sl.add_precondition_to_action(agent_name, action_name, 'assigned_truck', ('truck',))
                driverlog_sl.add_precondition_to_action(agent_name, action_name, 'assigned_package', ('obj',))

            driverlog_sl.add_agent_goal(agent_name, 'empty', (truck_name,))

        assigned_packages = set()
        for agent_name, goals in agent_goals.items():
            for goaltuple in goals:
                if goaltuple[0] != 'at':
                    continue
                obj_name = goaltuple[1][0]
                if obj_name not in self.instance_data['packages'] or obj_name in assigned_packages:
                    continue
                driverlog_sl.set_initial_value_for_new_fluent(agent_name, 'assigned_package', (obj_name,), True)
                assigned_packages.add(obj_name)

        for package_index, package_name in enumerate(self.instance_data['packages']):
            if package_name in assigned_packages:
                continue
            agent_name = self.instance_data['agents'][package_index % len(self.instance_data['agents'])]
            driverlog_sl.set_initial_value_for_new_fluent(agent_name, 'assigned_package', (package_name,), True)

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
            self.problem = self.add_social_law()
        return self.problem
