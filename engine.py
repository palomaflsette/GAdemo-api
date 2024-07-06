import sympy as sp
import random
import numpy as np
import matplotlib.pyplot as plt


def get_function():
    """Aceita uma função matemática do usuário."""
    func_str = input(
        "Insira a função em termos de x e y (por exemplo, 'sin(sqrt(x**2 + y**2))**2 - 0.5'): ")
    x, y = sp.symbols('x y')
    func = sp.sympify(func_str)
    return func, x, y


def execution_characteristics(num_evaluations, population_size, crossover_rate, mutation_rate, num_generations):
    characteristics = {
        "num_evaluations": num_evaluations,
        "population_size": population_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "num_generations": num_generations
    }
    return characteristics


def algorithm_characteristics(normalization_linear, elitism, steady_state, steady_state_no_duplicates):
    characteristics = {
        "normalization_linear": normalization_linear,
        "elitism": elitism,
        "steady_state": steady_state,
        "steady_state_no_duplicates": steady_state_no_duplicates
    }
    return characteristics


def crossover_type(one_point, two_point, uniform):
    characteristics = {
        "one_point": one_point,
        "two_point": two_point,
        "uniform": uniform
    }
    return characteristics


def evaluate_population(population, func, x, y):
    return [func.evalf(subs={x: ind[0], y: ind[1]}) for ind in population]


def select_parents(population, fitnesses, elitism):
    selected = []
    if elitism:
        sorted_population = [x for _, x in sorted(zip(fitnesses, population))]
        selected.extend(sorted_population[:2])
    while len(selected) < len(population):
        tournament = random.sample(list(zip(fitnesses, population)), 3)
        selected.append(max(tournament, key=lambda item: item[0])[1])
    return selected


def crossover(parent1, parent2, crossover_type):
    if crossover_type['one_point']:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    elif crossover_type['two_point']:
        point1, point2 = sorted(random.sample(range(1, len(parent1)), 2))
        return (parent1[:point1] + parent2[point1:point2] + parent1[point2:],
                parent2[:point1] + parent1[point1:point2] + parent2[point2:])
    elif crossover_type['uniform']:
        return ([(p1 if random.random() < 0.5 else p2) for p1, p2 in zip(parent1, parent2)],
                [(p1 if random.random() >= 0.5 else p2) for p1, p2 in zip(parent1, parent2)])
    else:
        return parent1, parent2


def mutate(individual, mutation_rate):
    return [gene + random.gauss(0, 1) * mutation_rate if random.random() < mutation_rate else gene for gene in individual]


def genetic_algorithm(func, x, y, exec_chars, alg_chars, cross_type):
    population = [[random.uniform(-10, 10), random.uniform(-10, 10)]
                  for _ in range(exec_chars['population_size'])]
    best_fitnesses = []

    for generation in range(exec_chars['num_generations']):
        fitnesses = evaluate_population(population, func, x, y)
        if alg_chars['normalization_linear']:
            min_fit, max_fit = min(fitnesses), max(fitnesses)
            fitnesses = [(f - min_fit) / (max_fit - min_fit)
                         for f in fitnesses]
        parents = select_parents(population, fitnesses, alg_chars['elitism'])
        next_generation = []
        while len(next_generation) < exec_chars['population_size']:
            parent1, parent2 = random.sample(parents, 2)
            offspring1, offspring2 = crossover(parent1, parent2, cross_type)
            next_generation.extend([mutate(offspring1, exec_chars['mutation_rate']),
                                    mutate(offspring2, exec_chars['mutation_rate'])])
        population = next_generation[:exec_chars['population_size']]
        best_fitness = max(fitnesses)
        best_fitnesses.append(best_fitness)
        best_individual = population[fitnesses.index(best_fitness)]
        print(
            f"Geração {generation + 1}: Melhor aptidão = {best_fitness}, Melhor indivíduo = {best_individual}")

    # Plotar o gráfico
    plt.plot(range(1, len(best_fitnesses) + 1), best_fitnesses)
    plt.xlabel('Geração')
    plt.ylabel('Melhor Aptidão')
    plt.title('Evolução da Melhor Aptidão')
    plt.show()

    return best_individual, best_fitness


def main():
    func, x, y = get_function()

    num_evaluations = int(
        input("Insira o número de avaliações (máx: 15000): "))
    population_size = int(input("Insira o tamanho da população (máx: 200): "))
    crossover_rate = float(input("Insira a taxa de crossover (%): "))
    mutation_rate = float(input("Insira a taxa de mutação (%): "))
    num_generations = int(input("Insira o total de rodadas (máx: 100): "))

    exec_chars = execution_characteristics(
        num_evaluations, population_size, crossover_rate, mutation_rate, num_generations)

    normalization_linear = input(
        "Deseja usar normalização linear? (s/n): ").lower() == 's'
    elitism = input("Deseja usar elitismo? (s/n): ").lower() == 's'
    steady_state = input("Deseja usar steady state? (s/n): ").lower() == 's'
    steady_state_no_duplicates = input(
        "Deseja usar steady state sem duplicados? (s/n): ").lower() == 's'

    alg_chars = algorithm_characteristics(
        normalization_linear, elitism, steady_state, steady_state_no_duplicates)

    one_point = input(
        "Deseja usar crossover de ponto único? (s/n): ").lower() == 's'
    two_point = input(
        "Deseja usar crossover de dois pontos? (s/n): ").lower() == 's'
    uniform = input("Deseja usar crossover uniforme? (s/n): ").lower() == 's'

    cross_type = crossover_type(one_point, two_point, uniform)

    best_individual, best_fitness = genetic_algorithm(
        func, x, y, exec_chars, alg_chars, cross_type)
    print(
        f"\nMelhor indivíduo encontrado: {best_individual} com aptidão {best_fitness}")


if __name__ == "__main__":
    main()
