import random
import json
if __name__ == '__main__':
    file = {}
    for i in range(1, 21):
        min_x = 0
        min_y = 0
        max_y = int(3+i/4)
        max_x = int(4+i/3)
        agents = [f'a{x}' for x in range(int(2+i/5))]
        file['agents'] = agents
        file['min_x'] = min_x
        file['max_x'] = max_x
        file['min_y'] = min_y
        file['max_y'] = max_y

        for a in agents:
            file[a] = {
                'init_x' : random.randint(0, max_x),
                'init_y' : random.randint(0, max_y),
                'goal_x' : random.randint(0, max_x),
                'goal_y' : random.randint(0, max_y),
            }
        with open(f"/home/evgeny/SocialLaws/up-social-laws/experimentation/numeric_problems/grid/pfile{i}.json", "w") as outfile:
            json.dump(file, outfile)


