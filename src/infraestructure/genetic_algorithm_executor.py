# Biblioteca para geração de números aleatórios (usada em mutação, crossover, etc.)
import random
import numpy as np  # Biblioteca para cálculos numéricos, como média e normalização
# DEAP fornece ferramentas para construção de Algoritmos Genéticos
from deap import base, creator, tools
import sympy as sp  # Biblioteca para manipulação simbólica de funções matemáticas
import concurrent.futures  # Permite execução paralela de experimentos

# Função auxiliar para determinar quantos indivíduos serão preservados pelo elitismo


def calculate_elite_count(population_size):
    """
    Retorna o número de indivíduos que serão preservados como elites com base no tamanho da população.
    - População <= 5: nenhum indivíduo é elitizado.
    - População entre 6 e 10: 1 indivíduo é elitizado.
    - População > 10: 2 indivíduos são elitizados.
    """
    if population_size <= 5:
        return 0
    elif 6 <= population_size <= 10:
        return 1
    else:
        return 2

# Classe principal para execução do Algoritmo Genético


class GeneticAlgorithmExecutor:
    def __init__(self):
        pass

    def get_function(self, func_str):
        x, y = sp.symbols('x y')
        func = sp.sympify(func_str)
        if len(func.free_symbols) == 1 and x in func.free_symbols:
            func = func + func.subs(x, y)
        return func, x, y

    def normalize_fitness(self, fitnesses, min_val, max_val):
        min_fit = min(fitnesses)
        max_fit = max(fitnesses)
        if min_fit == max_fit:
            return [min_val for _ in fitnesses]
        return [
            (min_val + (max_val - min_val) * (fit - min_fit) / (max_fit - min_fit))
            for fit in fitnesses
        ]

    def run_genetic_algorithm(self, func, exec_chars, cross_type):
        func, x, y = self.get_function(func)

        creator.create("Fitness", base.Fitness, weights=(
            1.0 if exec_chars.maximize else -1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()
        interval = exec_chars.interval

        toolbox.register("attr_float", random.uniform,
                         interval[0], interval[1])
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_float, n=2 if y else 1)
        toolbox.register("population", tools.initRepeat,
                         list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate_func, func=func, x=x, y=y)

        if cross_type.one_point:
            toolbox.register("mate", tools.cxOnePoint)
        elif cross_type.two_point:
            toolbox.register("mate", tools.cxTwoPoint)
        elif cross_type.uniform:
            toolbox.register("mate", tools.cxUniform, indpb=0.5)

        toolbox.register("mutate", self.mutate_within_bounds,
                         interval=interval, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=exec_chars.population_size)

        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        best_individuals = []

        for gen in range(exec_chars.num_generations):
            if exec_chars.normalize_linear:
                fitnesses = [ind.fitness.values[0] for ind in population]
                normalized_fitnesses = self.normalize_fitness(
                    fitnesses, exec_chars.normalize_min, exec_chars.normalize_max)
                for ind, norm_fit in zip(population, normalized_fitnesses):
                    ind.fitness.values = (norm_fit,)

            if exec_chars.steady_state or exec_chars.steady_state_without_duplicateds:
                gap = exec_chars.gap / 100
                gap = max(1, int(gap * len(population)))
                offspring = toolbox.select(population, gap)
                offspring = list(map(toolbox.clone, offspring))

                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < exec_chars.crossover_rate:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < exec_chars.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                invalid_ind = [
                    ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                if exec_chars.steady_state_without_duplicateds:
                    offspring = [
                        ind for ind in offspring if ind not in population]

                population.extend(offspring)
                population.sort(key=lambda ind: ind.fitness.values,
                                reverse=exec_chars.maximize)
                population = population[:exec_chars.population_size]

            else:
                elites = []
                if exec_chars.elitism:
                    elite_count = calculate_elite_count(len(population))
                    elites = tools.selBest(population, elite_count)

                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))

                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < exec_chars.crossover_rate:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < exec_chars.mutation_rate:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                invalid_ind = [
                    ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                population[:] = offspring

                if exec_chars.elitism and elites:
                    population.extend(elites)
                    population.sort(
                        key=lambda ind: ind.fitness.values, reverse=exec_chars.maximize)
                    population[:] = population[:exec_chars.population_size]

            best_individual = tools.selBest(population, 1)[0]
            best_individuals.append(
                (gen, best_individual[:], best_individual.fitness.values[0]))

        last_generation_values = [ind.fitness.values[0] for ind in population]
        return population, best_individuals, last_generation_values

    async def run_multiple_experiments(self, func, exec_chars, cross_type, num_experiments):
        best_experiment_values = []
        best_individuals_per_experiment = []
        best_values_per_generation = []
        last_generation_values_per_experiment = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.run_genetic_algorithm,
                                func, exec_chars, cross_type)
                for _ in range(num_experiments)
            ]

            for future in concurrent.futures.as_completed(futures):
                population, best_individuals, last_generation_values = future.result()

                best_experiment_values.append(max(ind[2] for ind in best_individuals))
                best_individuals_per_experiment.append(
                    [ind[1] for ind in best_individuals])
                best_values_per_generation.append(
                    [ind[2] for ind in best_individuals])
                last_generation_values_per_experiment.append(
                    last_generation_values)

        best_individuals_per_generation = [
            [
                best_individuals[gen]
                for best_individuals in best_individuals_per_experiment
            ]
            for gen in range(exec_chars.num_generations)
        ]

        average_fitness_best_individuals_per_generation = [
            float(np.mean([fit for fit in generation]))
            for generation in best_values_per_generation
        ]

        return (
            best_experiment_values,
            best_individuals_per_experiment,
            average_fitness_best_individuals_per_generation,
            best_values_per_generation,
            last_generation_values_per_experiment
        )

    def evaluate_func(self, individual, func, x, y=None):
        if y is not None:
            x_val, y_val = individual
            result = func.subs({x: x_val, y: y_val})
        else:
            x_val = individual[0]
            result = func.subs({x: x_val})
        return float(result),

    def mutate_within_bounds(self, individual, interval, mu, sigma, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(mu, sigma)
                individual[i] = max(
                    min(individual[i], interval[1]), interval[0])
        return individual,
