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
    print(text[start_id])

    end_id = start_id
    print('end')
    print(text[end_id])

    parentheses_opened = 1
    while end_id < len(text):
        if parentheses_opened == 0:
            break
        end_id += 1
        print(text[end_id])

        if text[end_id] == '(':
            parentheses_opened += 1
        if text[end_id] == ')':
            parentheses_opened -= 1
        print(parentheses_opened)
    return text[start_id:end_id]


class Parser:
    def __init__(self):
        self.agents = []
        self.objects = {}
        self.general_change = {}
        self.agent_change = {}
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

    def transform_fluentuples(self, init_str, extract_agent_tuples):
        if extract_agent_tuples:
            self.agent_fluenttuples = []
        inits = ''
        for line in init_str.split('\n'):
            line = line.replace('(', '').replace(')', '').replace('\n', '').replace('\t', '').strip()
            if len(line) < 1:
                continue
            usable = line.split(' ')
            if usable[0] == '':
                continue

            operator = value = None
            change = self.general_change
            for u in usable:
                if self.agent_name is not None and self.agent_name in u:
                    change = self.agent_change
            if any([skip in usable for skip in self.skip_fluents]):
                continue
            usable[0] = usable[0] if not usable[0] in change else change[usable[0]]
            fluenttuple = usable
            if extract_agent_tuples:
                if len(fluenttuple) < 2:
                    breakpoint()
                if len(fluenttuple[1]) > 0 and any([var in self.agents in var for var in fluenttuple[1]]):
                    self.agent_fluenttuples.append(fluenttuple)
            if inits != '':
                inits += ',\n'
            inits = inits + str(fluenttuple).replace('\'', '\"')
        return inits

    def get_objects_and_agents(self, base_objects_str):
        objects_str = ''
        agents_str = ''
        for line in base_objects_str.split('\n'):
            if ' - ' not in line:
                continue
            line = line.replace(' - ', ' ').replace('\t', '').replace('\n', '')
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

    def make_agent_fluents(self, fluent_convertion_dict):
        agent_fluents_str = ''
        agent_fluenttuples = {}
        for fluenttuple in self.agent_fluenttuples:
            operator = value = None
            if fluenttuple[0] in ['=', '>', '<']:
                operator = fluenttuple[0]
                value = fluenttuple[-1]
                fluenttuple = fluenttuple[1]
            new_fluent_name = fluenttuple[0]
            if fluenttuple[0] in fluent_convertion_dict:
                new_fluent_name = fluent_convertion_dict[new_fluent_name]
            params = []
            agent_name = None
            for p in fluenttuple[1]:
                if self.agent_name in p:
                    agent_name = p
                else:
                    params.append(p)
            if agent_name not in agent_fluenttuples:
                agent_fluenttuples[agent_name] = []
            if operator is None:
                agent_fluenttuples[agent_name].append([new_fluent_name, params])
            else:
                agent_fluenttuples[agent_name].append([operator, [new_fluent_name, params], value])
        counter = 0
        for agent_name in agent_fluenttuples:
            agent_fluents_str += f'\"{agent_name}\":{agent_fluenttuples[agent_name]}'
            counter += 1
            if counter != len(agent_fluenttuples):
                agent_fluents_str += ','
            agent_fluents_str += '\n'

        return agent_fluents_str.replace('\'', '\"')

    def parse_json(self, pathname, agents_type_name=None):
        self.agent_change = {}
        self.general_change = {}
        self.skip_fluents = []
        json_ver = '{'
        with open(pathname, 'r') as file:
            content = file.read()

        objects_str = extract_parenthesis_after('objects', content)
        init_str = extract_parenthesis_after('init', content)

        goal_str = extract_parenthesis_after('goal', content)
        metric_str = extract_parenthesis_after('metric', content)


        objs, agents = self.get_objects_and_agents(objects_str)

        json_ver += objs + ',\n'
        json_ver += agents
        json_ver += '],\n\n\"init_values\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(init_str, extract_agent_tuples=True, ) + '],\n\n'
        json_ver += self.make_agent_fluents({})
        json_ver += '\n},\n\"goals\": [\n'
        json_ver += self.transform_goal(goal_str) + '\n\n'
        json_ver += ']}'
        # if metric_str is not None:
        #    json_ver += self.extract_metric(metric_str)

        return json_ver


def redo():
    # Open the file for reading
    pathname = '/experimentation/numeric_problems/zenotravel/json/zenotravel/pfile2.json'
    with open(pathname, 'r') as file:
        content = file.read().replace('(', '[').replace('\'', '\"').replace(')', ']').replace(',]]', ']]')

    # Open the file for writing
    with open(pathname, 'w') as file:
        file.write(content)


if __name__ == '__main__':
    parser = Parser()
    parser.agent_type_name = 'sled'
    pathname = r'/home/evgeny/SocialLaws/up-social-laws/experimentation/numeric_problems/expedition/pddl/pfile1.pddl'
    print(parser.parse_json(pathname))
