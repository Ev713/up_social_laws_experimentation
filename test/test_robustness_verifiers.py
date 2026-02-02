from experimentation.problem_generators import problem_generator, expedition_generator
from up_social_laws import snp_to_num_strips

if __name__ == '__main__':

    problem_gen = expedition_generator.ExpeditionGenerator()
    problem_gen.instances_folder = 'experimentation/numeric_problems_instances/expedition/json'
    problem_gen.generate_problem('pfile1.json')
    problem_gen.add_social_law()
    problem = problem_gen.problem
    numeric_strips_problem = snp_to_num_strips.MultiAgentNumericStripsProblemConverter(problem).compile()
    print(numeric_strips_problem)
