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
        return None  # end_str not found

    # Return the substring between start_str and end_str
    return text[start_idx:end_idx]

class Parser:
    def __init__(self):
        self.agent_fluenttuples = []
        self.agent_name = 'plane'

    def transform_fluentuples(self, init_str, extract_agent_tuples):
        if extract_agent_tuples:
            self.agent_fluenttuples=[]
        inits = ''
        for line in init_str.split('\n'):
            line = line.replace('(', '').replace(')', '').replace('\n', '').replace('\t', '')
            if len(line) < 1:
                continue
            if inits != '':
                inits += ',\n'
            usable = line.split(' ')
            fluenttuple = [usable[0], [usable[1], ]] if len(usable) < 3 else [usable[0], [usable[1], usable[2]]]
            if extract_agent_tuples:
                flag = False
                if len(fluenttuple[1])==1:
                    if self.agent_name in fluenttuple[1][0]:
                        flag=True
                else:
                    if self.agent_name in fluenttuple[1][0]:
                        flag=True
                if flag:
                    self.agent_fluenttuples.append(fluenttuple)
                    continue
            inits = inits + str(fluenttuple).replace('\'', '\"')
        return inits

    def get_objects(self, objects_str):
        return_str = ''
        objects = {}
        for line in objects_str.split('\n'):
            if len(line) < 1:
                continue
            line = line.replace(' - ', ' ').replace('\t', '').replace('\n', '')
            line = line.split(' ')
            if line[1] + 's' not in objects:
                objects[line[1] + 's'] = [line[0]]
            else:
                objects[line[1] + 's'].append(line[0])
        for key in objects:
            objects[key] = sorted(objects[key])
        for key in objects:
            if return_str != '':
                return_str += ',\n'
            return_str += f'\"{key}\": ' + str(objects[key]).replace('\'', '\"')
        return return_str

    def make_agent_fluents(self, fluent_convertion_dict):
        agent_fluents_str = ''
        agent_fluenttuples = {}
        for fluenttuple in self.agent_fluenttuples:
            new_fluent_name = fluent_convertion_dict[fluenttuple[0]]
            params = []
            agent_name = None
            for p in fluenttuple[1]:
                if self.agent_name in p:
                    agent_name=p
                else:
                    params.append(p)
            if agent_name not in agent_fluenttuples:
                agent_fluenttuples[agent_name] = []
            agent_fluenttuples[agent_name].append([new_fluent_name, params])
        counter = 0
        for agent_name in agent_fluenttuples:
            agent_fluents_str+=f'\"{agent_name}\":{agent_fluenttuples[agent_name]}'
            counter+=1
            if counter!=len(agent_fluenttuples):
                agent_fluents_str+=','
            agent_fluents_str+='\n'

        return agent_fluents_str.replace('\'', '\"')


    def parse_json(self, number):
        pathname = r'C:\Users\foree\PycharmProjects\up-social-laws\experimentation\problems\all\unfactored\zenotravel\pfile'+str(number)+'.pddl'
        json_ver = '{'
        with open(pathname, 'r') as file:
            content = file.read()

        objects_str = extract_between(content, 'objects', '\n\n')
        init_str = extract_between(content, 'init', ')\n(')
        goal_str = extract_between(content, 'and', ')\n)')
        json_ver += self.get_objects(objects_str) + ',\n'
        num_of_agents = content.count('private')
        json_ver += ('\"agents": [')
        for i in range(num_of_agents):
            if i != 0:
                json_ver += ', '
            json_ver += '\"' + self.agent_name + str(i + 1) + '\"'
        json_ver += '],\n\n\"init_values\": {\n\"global\": [\n'
        json_ver += self.transform_fluentuples(init_str, extract_agent_tuples=True) + '],\n\n'
        json_ver+=self.make_agent_fluents({'at': 'aircraft_at', 'fuel-level':'fuel-level'})

        json_ver+='\n},\n\"goals\": [\n'
        json_ver += self.transform_fluentuples(goal_str, extract_agent_tuples=False) + '\n\n'
        json_ver += ']}'
        return json_ver.replace('\"at\"', '\"person_at\"')


def redo():
    # Open the file for reading
    pathname = '/home/evgeny/SocialLaws/up-social-laws/experimentation/problems/all/jsons/blocksworld/17-0.json'
    with open(pathname, 'r') as file:
        content = file.read().replace('(', '[').replace('\'', '\"').replace(')', ']').replace(',]]', ']]')

    # Open the file for writing
    with open(pathname, 'w') as file:
        file.write(content)


if __name__ == '__main__':
    p = Parser()
    for number in range(13, 24):
        pathname = r'C:\Users\foree\PycharmProjects\up-social-laws\experimentation\problems\all\jsons\zenotravel\pfile'+str(number)+'.json'
        with open(pathname, 'w') as file:
            file.write(p.parse_json(number))