import logging

from experimentation.problem_generators.expedition_generator import ExpeditionGenerator
from experimentation.problem_generators.market_trader_generator import MarketTraderGenerator
from experimentation.problem_generators.numeric_zenotravel_generator import NumericZenotravelGenerator
from up_social_laws.snp_to_num_strips import MultiAgentNumericStripsProblemConverter

long_dash = "-----------------"

def test_expedition_generation():
    prob_gen = ExpeditionGenerator()
    prob_gen.instances_folder = "numeric_problems_instances/expedition/json"
    for instance_id in range(1, 21):
        instance_name = f'pfile{instance_id}.json'
        prob = prob_gen.generate_problem(instance_name)
        logging.debug(f"GENERATING EXPEDITION PROBLEM NO. {instance_id} RESULTED IN:\n{long_dash}\n{prob}\n{long_dash}\n")

def test_numeric_zenotrvel_generation():
    prob_gen = NumericZenotravelGenerator()
    prob_gen.instances_folder = "numeric_problems_instances/zenotravel/json"
    for instance_id in range(1, 21):
        instance_name = f'pfile{instance_id}.json'
        prob = prob_gen.generate_problem(instance_name)
        logging.debug(
            f"GENERATING EXPEDITION PROBLEM NO. {instance_id} RESULTED IN:\n{long_dash}\n{prob}\n{long_dash}\n")

def test_markettrader_generation():
    prob_gen = MarketTraderGenerator()
    prob_gen.instances_folder = "numeric_problems_instances/markettrader/json"
    for instance_id in range(1, 21):
        instance_name = f'pfile{instance_id}.json'
        prob = prob_gen.generate_problem(instance_name)
        logging.debug(
            f"GENERATING EXPEDITION PROBLEM NO. {instance_id} RESULTED IN:\n{long_dash}\n{prob}\n{long_dash}\n")

def get_expedition_problems():
    problems = []
    prob_gen = ExpeditionGenerator()
    prob_gen.instances_folder = "numeric_problems_instances/expedition/json"
    for instance_id in range(1, 21):
        instance_name = f'pfile{instance_id}.json'
        problems.append(prob_gen.generate_problem(instance_name))
    return problems

def get_numeric_zenotravel_problems():
    problems = []
    prob_gen = NumericZenotravelGenerator()
    prob_gen.instances_folder = "numeric_problems_instances/zenotravel/json"
    for instance_id in range(1, 21):
        instance_name = f'pfile{instance_id}.json'
        problems.append(prob_gen.generate_problem(instance_name))
    return problems

def get_markettrader_problems():
    problems = []
    prob_gen = MarketTraderGenerator()
    prob_gen.instances_folder = "numeric_problems_instances/markettrader/json"
    for instance_id in range(1, 21):
        instance_name = f'pfile{instance_id}.json'
        problems.append(prob_gen.generate_problem(instance_name))
    return problems

def test_convert_problems_to_num_strips():
    problems = get_expedition_problems()
    problems += get_numeric_zenotravel_problems()
    problems += get_markettrader_problems()

    for p in problems:
        try:
            MultiAgentNumericStripsProblemConverter(p).compile()
            logging.info(f"Convertion of {p.name} was successful")
        except Exception as e:
            logging.info(f"Convertion of {p.name} failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_convert_problems_to_num_strips()