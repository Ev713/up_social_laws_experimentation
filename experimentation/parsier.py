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
        end_idx = len(text)-1  # end_str not found

    # Return the substring between start_str and end_str
    return text[start_idx:end_idx]


class Parser:
    def __init__(self):
        self.general_change = {}
        self.agent_change = {}
        self.agent_fluenttuples = []
        self.agent_name = None
        self.agent_type_name = None
        self.skip_fluents = []

    def transform_goal(self, goal_str):
        goal_str = goal_str.replace('\n', ' ').replace('\t', ' ')
        goal_words = [word.strip() for word in goal_str.split()]
        i = 0
        while True:
            raise NotImplementedError

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

                if len(fluenttuple[1]) > 0 and any([self.agent_name in var for var in fluenttuple[1]]):
                    self.agent_fluenttuples.append(fluenttuple)
            if inits != '':
                inits += ',\n'
            inits = inits + str(fluenttuple).replace('\'', '\"')
        return inits
    def get_objects_and_agents(self, base_objects_str):
        objects_str = ''
        agents_str = ''
        objects = {}
        for line in base_objects_str.split('\n'):
            if ' - ' not in line:
                continue
            line = line.replace(' - ', ' ').replace('\t', '').replace('\n', '')
            line = line.split(' ')
            if line[1] + 's' not in objects:
                objects[line[1] + 's'] = [line[0]]
            else:
                objects[line[1] + 's'].append(line[0])
        for key in objects:
            objects[key] = sorted(objects[key])

        agents = None
        if not(self.agent_name == None and self.agent_type_name ==None):
            for agent_name in [self.agent_type_name, self.agent_type_name + 's']:
                if agent_name in objects:
                    agents = objects[agent_name]
                    objects.pop(agent_name, None)
            for key in objects:
                if objects_str != '':
                    objects_str += ',\n'
                objects_str += f'\"{key}\": ' + str(objects[key]).replace('\'', '\"')
            if agents is None:
                return objects_str
            agents_str += ('\"agents": [')
            for i, a in enumerate(agents):
                if i != 0:
                    agents_str += ', '
                agents_str += '\"' + a + '\"'
        return objects_str, agents_str

    def extract_metric(self, metric_str):
        metric_str = metric_str.replace('(', '').replace(')', '')
        metric_str = metric_str.split()

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
        self.agent_change = {'located': 'aircraft-loc'}
        self.general_change = {'located': 'person-loc'}
        self.skip_fluents = ['total-fuel-used']
        json_ver = '{'
        with open(pathname, 'r') as file:
            content = file.read()

        objects_str = extract_between(content, 'objects', ')')
        init_str = extract_between(content, 'init', ')\n(')
        goal_str = extract_between(content, ':goal', '\n\n\n')
        metric_str = content.split('metric')
        if len(metric_str) > 1:
            metric_str = metric_str[1]
        else:
            metric_str = None

        objs, agents = self.get_objects_and_agents(objects_str)

        json_ver += objs + ',\n'
        json_ver += agents
        json_ver += '],\n\n\"init_values\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(init_str, extract_agent_tuples=False,) + '],\n\n'
        #json_ver += self.make_agent_fluents({'located': 'aircraft_at', 'fuel': 'fuel'})
        json_ver += '\n},\n\"goals\": [\n'
        json_ver += self.transform_goal(goal_str) + '\n\n'
        json_ver += ']}'
        # if metric_str is not None:
        #    json_ver += self.extract_metric(metric_str)

        return json_ver.replace('\"at\"', '\"person_at\"')


def redo():
    # Open the file for reading
    pathname = '/home/evgeny/SocialLaws/up-social-laws/experimentation/numeric_problems/all/json/zenotravel/pfile2.json'
    with open(pathname, 'r') as file:
        content = file.read().replace('(', '[').replace('\'', '\"').replace(')', ']').replace(',]]', ']]')

    # Open the file for writing
    with open(pathname, 'w') as file:
        file.write(content)


if __name__ == '__main__':
    parser = Parser()
    pathname = r'C:\Users\foree\PycharmProjects\up_social_laws_experimentation\experimentation\problems\numeric_problems\unfactored\block-grouping\pfile1.pddl'
    print(Parser().parse_json(pathname))

