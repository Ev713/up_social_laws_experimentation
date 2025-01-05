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
        self.agent_name = None
        self.agent_type_name = None
        self.skip_fluents = []

    def transform_goal(self, goal_str):
        json_goal_str = ''
        goal_str = goal_str.replace('\n', ' ').replace('\t', ' ')
        goal_words = [word.strip() for word in goal_str.split()]
        skip_to = 0
        for i, start_word in enumerate(goal_words):
            if 'and' in start_word or len(start_word) == 0 or i < skip_to:
                continue
            if ')' in start_word:
                break
            if '(' in start_word:
                j = i + 1
                while ')' not in goal_words[j - 1]:
                    j += 1
                skip_to = j
                vars = goal_words[i + 1:j]
                vars[-1] = vars[-1].replace(')', '')
                fluenttuple = [start_word.replace('(', ''), vars]
                json_goal_str += str(fluenttuple) + '\n'
            else:
                continue
        return json_goal_str

    def transform_fluentuples(self, init_str, extract_agent_tuples, ):
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
                expression = [operator]+ expression+[value]
            if len(expression)==1:
                expression = expression[0]
            if agent is not None:
                if agent not in self.agent_fluenttuples:
                    self.agent_fluenttuples[agent] = []
                self.agent_fluenttuples[agent].append(expression)
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
        self.all_objects = []
        for obj_type in self.objects:
            self.all_objects+=self.objects[obj_type]
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

    def parse_json(self, pathname, agents_type_name=None):
        self.agent_changer = {}
        self.general_changer = {}
        self.skip_fluents = []
        json_ver = '{'
        with open(pathname, 'r') as file:
            content = file.read()

        objects_str = extract_parenthesis_after('objects', content).strip()
        init_str = extract_parenthesis_after('init', content).strip()

        goal_str = extract_parenthesis_after('goal', content)
        goal_str = extract_parenthesis_after('and', goal_str).strip()

        metric_str = extract_parenthesis_after('metric', content)

        objs, agents = self.get_objects_and_agents(objects_str)

        json_ver += objs + ',\n'
        json_ver += agents
        json_ver += '],\n\n\"init_values\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(init_str, extract_agent_tuples=True) + ']'
        json_ver += self.make_agent_fluents()
        json_ver += '\n},\n\"goals\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(goal_str, extract_agent_tuples=True) + ']'
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
        parser.agent_type_name = {'expedition': 'sled', 'markettrader': 'camel', 'zenotravel': 'aircraft'}[domain]
        folder = r'C:\Users\foree\PycharmProjects\up_social_laws_experimentation\experimentation\numeric_problems'+'\\'+domain
        for i in range(1, 21):
            pathname = folder+r'\pddl\pfile'+str(i)+'.pddl'
            f = open(folder+r'\json\pfile'+str(i)+'.json', "w")
            f.write(parser.parse_json(pathname))
            f.close()