import random
import numpy as np
import sympy as sp
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# Função para aceitar funções matemáticas do usuário


def get_function(func_str):
    """Aceita uma função matemática do usuário."""
    x, y = sp.symbols('x y')
    func = sp.sympify(func_str)
    return func, x, y

# Função para definir características da execução


def execution_characteristics(num_evaluations, population_size, crossover_rate, mutation_rate, num_generations, maximize, interval):
    characteristics = {
        "num_evaluations": num_evaluations,
        "population_size": population_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "num_generations": num_generations,
        "maximize": maximize,
        "interval": interval
    }
    return characteristics

# Função para definir o tipo de crossover


def crossover_type(one_point, two_point, uniform):
    characteristics = {
        "one_point": one_point,
        "two_point": two_point,
        "uniform": uniform
    }
    return characteristics

# Função para avaliar a função do usuário
def evaluate_func(individual, func, x, y):
    # Converte o indivíduo em variáveis x e y
    x_val, y_val = individual
    # Avalia a função do usuário
    result = func.subs({x: x_val, y: y_val})
    return float(result),  # Retorna como uma tupla contendo um float

# Função principal para executar o algoritmo genético


def run_genetic_algorithm(func_str, exec_chars, cross_type):
    func, x, y = get_function(func_str)

    # Definir o tipo de fitness (maximização ou minimização)
    creator.create("Fitness", base.Fitness, weights=(
        1.0 if exec_chars["maximize"] else -1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    interval = exec_chars["interval"]
    toolbox.register("attr_float", random.uniform, interval[0], interval[1])
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_func, func=func, x=x, y=y)
    toolbox.register("mate", tools.cxOnePoint if cross_type["one_point"] else (
        tools.cxTwoPoint if cross_type["two_point"] else tools.cxUniform))
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=exec_chars["population_size"])

    # Estatísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Log
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Evolução
    for gen in range(exec_chars["num_generations"]):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < exec_chars["crossover_rate"]:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < exec_chars["mutation_rate"]:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    return population, logbook


# Exemplo de uso
func_str = "0.5 - (sin(sqrt(x**2 + y**2))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2"
exec_chars = execution_characteristics(num_evaluations=4000, population_size=100, crossover_rate=0.65,
                                       mutation_rate=0.08, num_generations=50, maximize=True, interval=(0, 4000))
cross_type = crossover_type(one_point=True, two_point=False, uniform=False)

population, logbook = run_genetic_algorithm(func_str, exec_chars, cross_type)

# Função para plotar os resultados


def plot_results(logbook, num_generations):
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_maxs = logbook.select("max")
    fit_avgs = logbook.select("avg")
    fit_stds = logbook.select("std")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_mins, label="Minimum Fitness")
    plt.plot(gen, fit_maxs, label="Maximum Fitness")
    plt.plot(gen, fit_avgs, label="Average Fitness")
    plt.fill_between(gen, np.array(fit_avgs) - np.array(fit_stds),
                     np.array(fit_avgs) + np.array(fit_stds), alpha=0.2, color='g')
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="best")
    plt.grid()
    plt.title("Fitness over Generations")
    # Ajusta o intervalo do eixo x para corresponder ao número de gerações
    plt.xlim(0, num_generations)
    plt.show()


# Plotar os resultados
plot_results(logbook, exec_chars["num_generations"])
