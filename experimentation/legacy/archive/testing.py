import os
import random
import json


def generate_instance(cities, goods, num_agents, avg_price, price_variance, cash, capacity, target_cash):
    # Initialize the instance structure
    instance = {
        "market": cities,
        "goods": goods,
        "agents": [f"camel{i}" for i in range(num_agents)],
        "init_values": {
            "global": [],
            "camel0": [],
        },
        "goals": {
            "global": [],
        }
    }

    # Generate prices and on-sale quantities for each good in each city
    for good in goods:
        cities_sell = random.sample(cities, max(1, int(len(cities)/2/len(goods))))
        for city in cities:
            price = round(avg_price + random.uniform(-price_variance, price_variance))
            sell_price = round(price * random.uniform(0.75, 2.0))
            instance["init_values"]["global"].append(["=", ["sellprice", [good, city]], str(sell_price)])

            if city in cities_sell:
                on_sale = random.randint(3, 15)
            else:
                continue
            instance["init_values"]["global"].append(["=", ["price", [good, city]], str(price)])
            instance["init_values"]["global"].append(["=", ["on-sale", [good, city]], str(on_sale)])

    # Generate drive costs between cities
    for city1 in cities:
        for city2 in cities:
            if city1 != city2:
                drive_cost = round(random.uniform(1.0, 5))
                instance["init_values"]["global"].append(["=", ["drive-cost", [city1, city2]], str(drive_cost)])

    # Initialize agents
    for i in range(num_agents):
        camel_name = f"camel{i}"
        instance["init_values"][camel_name] = [
            ["=", ["bought", [good]], "0"] for good in goods
        ]
        instance["init_values"][camel_name].append(["at", [random.choice(cities)]])
        instance["init_values"][camel_name].append(["=", ["cash", []], str(cash)])
        instance["init_values"][camel_name].append(["=", ["capacity", []], str(capacity)])

        # Add drive permissions
        for city1 in cities:
            for city2 in cities:
                if city1 != city2 and random.randint(1, 7) != 1:
                    instance["init_values"][camel_name].append(["can-drive", [city1, city2]])

        # Add goals for the agent
        instance["goals"][camel_name] = [[">=", ["cash", []], str(target_cash)]]

    return instance


# Customize parameters
all_cities = ["Athens", "Venice", "Moscow", "Nice", "Stockholm", "Jurmala", "Dublin", "Panemune", "Ogre", "Daugai",
              "Oslo",
              "Amsterdam", "Cesis", "Copenhagen", "Jelgava", "Dusetos", "Obeliai", "Riga", "Seda", "Sough",
              "Gelgaudiskis",
              "StPetersburg", "Rakvere", "Palanga", "Tartu", "Berlin", "Madrid", "Jieznas", "Mariehamn", "Kavarskas",
              "Tukums", "Douglas", "Brussels", "Lisbon", "Valencia", "Edinburgh", "Neringa", "Kaunas", "Simnas",
              "Paris",
              "KudirkosNaumiestis", "Narva", "Valmiera", "Salaspils", "Longyearbyen", "London", "Cardiff", "Tallinn",
              "Kadagopya", "Hamburg", "Vienna", "Thule", "Daugavpils", "Bonn", "Mandres", "Vilkija", "Rome", "Torshavn",
              "Viljandi"]
all_goods = ["ExpensiveRugs", "GummyBears", "Copper", "LaminateFloor", "Gold", "Footballs", "Platinum", "Water",
             "Kittens",
             "Food", "Computers", "DVDs", "TuringMachines", "Minerals", "Cars", "Coffee", "Cattle"]
print(f'Total cities: {len(all_cities)}')
print(f'Total goods: {len(all_goods)}')

for i in range(1, 21):
    print(f'\npfile{i}')
    num_agents = max(2, 2 + int(i / 7) + random.randint(-1, 1))
    print(f'Num. of agents: {num_agents}')
    avg_price = 3
    print(f'Average price: {avg_price}')
    price_variance = 2.0
    print(f'Price variance: {price_variance}')
    cash = 10
    print(f'Cash: {cash}')
    capacity = 5
    print(f'Capacity: {capacity}')
    target_cash = int(15 + 50*(i/20)*(0.5+random.random()))
    print(f'Target cash: {target_cash}')
    num_cities = 3 + int((len(all_cities) - 45) / 20 * i)
    print(f'Num cities: {num_cities}')
    num_goods = 1 + int((len(all_goods) - 10) / 20 * i)
    print(f'Num of goods: {num_goods}')
    cities = random.sample(all_cities, num_cities)
    goods = random.sample(all_goods, num_goods)
    random_instance = generate_instance(cities, goods, num_agents, avg_price, price_variance, cash, capacity,
                                        target_cash)

    directory = "./numeric_problems_instances/generated_json/"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"pfile{i}.json")
    with open(f"numeric_problems_instances/markettrader/generated_json/pfile{i}.json", "w") as file:
        json.dump(random_instance, file, indent=4)

