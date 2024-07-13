import random
import numpy as np
from deap import base, creator, tools, algorithms
import sympy as sp
import matplotlib.pyplot as plt


class GeneticAlgorithmExecutor:
    def __init__(self):
        pass

    def get_function(self, func_str):
        """Aceita uma função matemática do usuário."""
        x, y = sp.symbols('x y')
        func = sp.sympify(func_str)
        return func, x, y

    def run_genetic_algorithm(self, func, exec_chars, cross_type):
        func, x, y = self.get_function(func)
        
        # Definir o tipo de fitness (maximização ou minimização)
        creator.create("Fitness", base.Fitness, weights=(
            1.0 if exec_chars.maximize else -1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()
        interval = exec_chars.interval
        toolbox.register("attr_float", random.uniform,
                        interval[0], interval[1])
        toolbox.register("individual", tools.initRepeat,
                        creator.Individual, toolbox.attr_float, n=2)
        toolbox.register("population", tools.initRepeat,
                        list, toolbox.individual)

        toolbox.register("evaluate", self.evaluate_func, func=func, x=x, y=y)
        toolbox.register("mate", tools.cxOnePoint if cross_type.one_point else (
            tools.cxTwoPoint if cross_type.two_point else tools.cxUniform))
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=exec_chars.population_size)

        # Avaliar a população inicial
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Estatísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Log
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + stats.fields

        # Armazenar melhores indivíduos e seus valores de fitness por geração
        best_individuals = []

        # Evolução
        for gen in range(exec_chars.num_generations):
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

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population[:] = offspring
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Armazena o melhor indivíduo de cada geração
            best_individual = tools.selBest(population, 1)[0]
            best_individuals.append(
                (gen, best_individual, best_individual.fitness.values[0]))

        return population, logbook, best_individuals

    def run_multiple_experiments(self, func, exec_chars, cross_type, num_experiments):
        best_experiment_values = []
        best_individuals_per_generation = []

        for exp in range(num_experiments):
            print(f"\nRodando experimento {exp + 1}/{num_experiments}")
            _, logbook, best_individuals = self.run_genetic_algorithm(
                func, exec_chars, cross_type)
            best_experiment_values.append(best_individuals[-1][2])
            if exp == 0:
                for gen, best_ind, best_fit in best_individuals:
                    best_individuals_per_generation.append([best_fit])
            else:
                for i, (gen, best_ind, best_fit) in enumerate(best_individuals):
                    best_individuals_per_generation[i].append(best_fit)

        mean_best_individuals_per_generation = np.mean(
            best_individuals_per_generation, axis=1)

        return best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation

    def evaluate_func(self, individual, func, x, y):
        x_val, y_val = individual
        result = func.subs({x: x_val, y: y_val})
        return float(result),  # Retorna como uma tupla contendo um float
