OPERATORS = ['=', '>=', '<=', '>', '<']


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_between(text, start_str, end_str):
    # Find the starting position of start_str
    start_idx = text.find(start_str)
    if start_idx == -1:
        return None  # start_str not found

    # Adjust to get position after start_str
    start_idx += len(start_str)

    # Find the ending position of end_str, starting from the end of start_str
    end_idx = text.find(end_str, start_idx)
    if end_idx == -1:
        end_idx = len(text) - 1  # end_str not found

    # Return the substring between start_str and end_str
    return text[start_idx:end_idx]


def extract_parenthesis_after(word, text):
    start_id = text.find(word)
    if start_id == -1:
        return ''
    start_id += len(word)
    end_id = start_id
    parentheses_opened = 1
    while end_id < len(text):
        if parentheses_opened == 0:
            break
        end_id += 1
        if text[end_id] == '(':
            parentheses_opened += 1
        if text[end_id] == ')':
            parentheses_opened -= 1
    return text[start_id:end_id]


class Parser:
    def __init__(self):
        self.all_objects = None
        self.agents = []
        self.objects = {}
        self.general_changer = {}
        self.agent_changer = {}
        self.agent_fluenttuples = []
        self.agent_type_name = None
        self.skip_fluents = []
        self.copy_to_all = []
        self.is_expedition = False
        self.distribute_fluent_in_goal = None

    def transform_fluentuples(self, init_str, extract_agent_tuples, is_goal):
        if extract_agent_tuples:
            self.agent_fluenttuples = {}
        inits = ''
        lines = init_str.split('\n')
        for line in lines:
            line = ' '.join(line.replace('(', '').replace(')', '').replace('\n', '').replace('\t', '').strip().split())
            if len(line) < 1:
                continue
            words = line.split(' ')
            if words[0] == '':
                continue
            operator = None
            value = None
            agent = None
            for word in words:
                if word in self.agents:
                    if agent is not None:
                        raise Exception('Multiple agents in 1 fluent expression')
                    words.pop(words.index(word))
                    agent = word
            if agent is not None:
                changer = self.agent_changer
            else:
                changer = self.general_changer
            words = [word if word not in changer else changer[word] for word in words]
            vars = [word in self.all_objects for word in words]
            if len(words) >= 2 and not vars[0] and not vars[1]:
                operator = words[0]
                words = words[1:]
                vars = vars[1:]
                if is_number(words[-1]):
                    value = words[-1]
                    words = words[:-1]
                    vars = vars[:-1]
                    if (self.distribute_fluent_in_goal is not None) and words[0] in \
                            self.distribute_fluent_in_goal and is_goal:
                        value = str(float(value)/len(self.agents))
            if words[0] in self.skip_fluents:
                continue
            expression = []
            fluents_num = 0
            i = 0
            fluent_expression = None
            fluent_vars = []
            while i < len(words):
                if not vars[i]:
                    if fluent_expression is not None:
                        expression.append([fluent_expression, fluent_vars])
                        fluent_vars = []
                    fluents_num += 1
                    fluent_expression = words[i]
                else:
                    fluent_vars.append(words[i])
                i += 1
            expression.append([fluent_expression, fluent_vars])
            if operator is not None:
                expression = [operator] + expression + [value]
            if len(expression) == 1:
                expression = expression[0]
            agents = [agent]
            skip_adding_global = agent is not None
            if words[0] in self.copy_to_all:
                agents = self.agents
                skip_adding_global = True

            for agent in agents:
                if agent is not None:
                    if agent not in self.agent_fluenttuples:
                        self.agent_fluenttuples[agent] = []
                    self.agent_fluenttuples[agent].append(expression)
            if skip_adding_global:
                continue

            if inits != '':
                inits += ',\n'
            inits = inits + str(expression).replace('\'', '\"')
        return inits

    def get_objects_and_agents(self, base_objects_str):
        objects_str = ''
        agents_str = ''
        for line in base_objects_str.split('\n'):
            if ' - ' not in line:
                continue
            line = line.replace(' - ', ' ').replace('\t', '').replace('\n', '').strip()
            line = line.split(' ')
            obj_type = line[-1]
            if obj_type not in self.objects:
                self.objects[obj_type] = line[:-1]
            else:
                self.objects[obj_type] += line[:-1]
        for key in self.objects:
            self.objects[key] = sorted(self.objects[key])

        if self.agent_type_name in self.objects:
            self.agents = self.objects[self.agent_type_name]
            self.objects.pop(self.agent_type_name, None)
        for obj_type in self.objects:
            self.objects[obj_type] = list(set(self.objects[obj_type]))
        self.all_objects = []
        for obj_type in self.objects:
            self.all_objects += self.objects[obj_type]
        for key in self.objects:
            if objects_str != '':
                objects_str += ',\n'
            objects_str += f'\"{key}\": ' + str(self.objects[key]).replace('\'', '\"')
        agents_str += ('\"agents": [')
        for i, a in enumerate(self.agents):
            if i != 0:
                agents_str += ', '
            agents_str += '\"' + a + '\"'
        return objects_str, agents_str

    def make_agent_fluents(self):
        agent_fluents_str = ''
        for agent_name in self.agent_fluenttuples:
            agent_fluents_str += ','
            agent_fluents_str += '\n'
            agent_fluents_str += f'\"{agent_name}\":{self.agent_fluenttuples[agent_name]}'

        return agent_fluents_str.replace('\'', '\"')

    def parse_json(self, pathname, agents_type_name=None, add_agents=None):
        json_ver = '{'
        with open(pathname, 'r') as file:
            content = file.read()

        if add_agents is not None:
            self.objects[self.agent_type_name] = add_agents
        objects_str = extract_parenthesis_after('objects', content).strip()
        init_str = extract_parenthesis_after('init', content).strip()

        goal_str = extract_parenthesis_after('goal', content)
        goal_str = extract_parenthesis_after('and', goal_str).strip()

        metric_str = extract_parenthesis_after('metric', content)

        objs, agents = self.get_objects_and_agents(objects_str)

        json_ver += objs + ',\n'
        json_ver += agents
        json_ver += '],\n\n\"init_values\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(init_str, extract_agent_tuples=True, is_goal=False) + ']'
        for agent in self.agents:
            if agent not in self.agent_fluenttuples:
                self.agent_fluenttuples[agent] = list(self.agent_fluenttuples.values())[0].copy()
        json_ver += self.make_agent_fluents()
        json_ver += '\n},\n\"goals\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(goal_str, extract_agent_tuples=True, is_goal=True) + ']'
        json_ver += self.make_agent_fluents()
        json_ver += '}}'
        # if metric_str is not None:
        #    json_ver += self.extract_metric(metric_str)

        return json_ver


if __name__ == '__main__':
    domains = [
        #    'expedition',
        'markettrader',
        #    'zenotravel'
    ]
    for domain in domains:
        parser = Parser()
        added_agents = [[] for _ in range(21)]
        if domain == 'zenotravel':
            parser.agent_changer = {'located': 'aircraft-loc'}
            parser.general_changer = {'located': 'person-loc'}
            parser.skip_fluents = ['total-fuel-used']
            parser.copy_to_all = []
            parser.agent_type_name = 'aircraft'
            added_agents[1] = ['plane2']
            added_agents[2] = ['plane2']

        elif domain == 'markettrader':
            parser.agent_type_name = 'camel'
            parser.skip_fluents = ['fuel-used', 'fuel']
            parser.copy_to_all = ['cash', 'capacity', 'bought', 'at', 'can-drive']
            added_agents = [[f'camel{k}' for k in range(1, int(i / 5) + 1)] for i in range(4, 25)]
            parser.distribute_fluent_in_goal = ['cash']
        elif domain == 'expedition':
            parser.agent_type_name = 'sled'
            added_agents = [[f's{k + 1}' for k in range(1, int(i / 5) + 1)] for i in range(-1, 20)]

        folder = './numeric_problems_instances/' + domain
        for i in range(1, 21):
            pathname = folder + '/pddl/pfile' + str(i) + '.pddl'
            f = open(folder + r'/json/pfile' + str(i) + '.json', "w")
            f.write(parser.parse_json(pathname, add_agents=added_agents[i]))
            f.close()
