import json

import unified_planning
from unified_planning.model import InstantaneousAction
from unified_planning.model.multi_agent import Agent
from unified_planning.shortcuts import *

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


OPERATORS = {
    '=': Equals,
    '>=': GE,
    '>': GT,
    '<=': LE,
    '<': LT
}


class NoSocialLawException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class ProblemGenerator:
    def __init__(self):
        self.obj_type = {}
        self.problem = None
        self.instance_data = None
        self.instances_folder = ''

    def generate_problem(self, file_name, sl=False):
        pass

    def add_social_law(self):
        raise NoSocialLawException

    def load_instance_data(self, instance_name):
        json_file_path = self.instances_folder + '/' + instance_name
        with open(json_file_path, 'r') as file:
            self.instance_data = json.load(file)
            return self.instance_data

    def load_objects(self, json_types, obj_types, remember=True):
        for i, obj_type in enumerate(obj_types):
            name = json_types[i]
            self.problem.add_objects(list(map(lambda x: unified_planning.model.Object(x, obj_type),
                                              self.instance_data[name])))
        if remember:
            self.remember_obj_types(json_types, obj_types)

    def load_agents(self):
        for agent_name in self.instance_data['agents']:
            self.problem.add_agent(Agent(agent_name, self.problem))

    def set_init_values(self):
        for key in self.instance_data['init_values']:
            if key == 'global':
                for fluentuple in self.instance_data['init_values'][key]:
                    fluent = self.problem.ma_environment.fluent(fluentuple[0])
                    params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
                    self.problem.set_initial_value(fluent(*params), True)

            else:
                agent = self.problem.agent(key)
                for fluentuple in self.instance_data['init_values'][key]:
                    fluent = fluentuple[0]
                    self.problem.set_initial_value(Dot(agent, agent.fluent(fluent)), True)

    def set_goals(self):
        agent_index = 0
        num_of_agents = len(self.problem.agents)
        for goaltuple in self.instance_data['goals']:
            agent = self.problem.agents[agent_index]
            agent_index = (agent_index + 1) % num_of_agents
            fluent = self.problem.ma_environment.fluent(goaltuple[0])
            params = (unified_planning.model.Object(v, self.obj_type[v]) for v in goaltuple[1])
            agent.add_public_goal(fluent(*params))

    def remember_obj_types(self, json_types, obj_types):
        self.obj_type = {}
        for i, json_type_name in enumerate(json_types):
            for obj_name in self.instance_data[json_type_name]:
                self.obj_type[obj_name] = obj_types[i]


class NumericProblemGenerator(ProblemGenerator):
    def __init__(self):
        super().__init__()
        self.agent_type_name = None

    def set_init_values(self):
        for key in self.instance_data['init_values']:
            for fluentuple in self.instance_data['init_values'][key]:
                value = True
                if fluentuple[0] in OPERATORS:
                    value = float(fluentuple[-1])
                    if value % 1 == 0:
                        value = int(value)
                    fluentuple = fluentuple[1]

                if key == 'global':
                    fluent = self.problem.ma_environment.fluent(fluentuple[0])
                    params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
                    self.problem.set_initial_value(fluent(*params), value)
                else:
                    agent = self.problem.agent(key)
                    fluent = agent.fluent(fluentuple[0])
                    params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
                    agent = self.problem.agent(key)
                    self.problem.set_initial_value(Dot(agent, fluent(*params)), value)

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

    def create_fluent_expression(self, fluentuple, agent):
        if is_number(str(fluentuple)):
            return float(fluentuple)
        if agent is None:
            fluent = self.problem.ma_environment.fluent(fluentuple[0])
        else:
            fluent = agent.fluent(fluentuple[0])
        params = (unified_planning.model.Object(v, self.obj_type[v]) for v in fluentuple[1])
        return fluent(*params)
