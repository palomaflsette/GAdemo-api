import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import sympy as sp
import pandas as pd
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

from domain.crossover_type import CrossoverType
from domain.execution_characteristics import ExecutionCharacteristics
from quarentine.function import Function
from infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor


def main():
    # Função matemática do usuário
    func_str = "x**2"

    # Características da execução
    exec_chars = ExecutionCharacteristics(
        num_generations=5,
        population_size=5,
        crossover_rate=1.0,
        mutation_rate=1.0,
        maximize=True,
        interval=(0, 10),
        crossover_type=CrossoverType(
            one_point=True, two_point=False, uniform=False)
    )

    # Instanciar o executor do algoritmo genético
    executor = GeneticAlgorithmExecutor()

    # Executar o algoritmo genético
    best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation = executor.run_multiple_experiments(
        func_str, exec_chars, exec_chars.crossover_type, num_experiments=4
    )

    # Exibir os resultados
    print("\nMelhores valores de cada experimento:")
    print(best_experiment_values)

    # Criando a tabela com os melhores indivíduos por geração
    df = pd.DataFrame(best_individuals_per_generation, columns=[
                      f'Exp {i+1}' for i in range(4)])
    df['Média'] = mean_best_individuals_per_generation

    print("\nTabela dos melhores indivíduos por geração:")
    print(df)

    # Plotando a curva da média dos melhores indivíduos por geração
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(mean_best_individuals_per_generation) + 1), mean_best_individuals_per_generation,
             marker='o', label="Média dos Melhores por Geração")
    plt.xlabel("Geração")
    plt.ylabel("Média da Aptidão")
    plt.legend(loc="best")
    plt.grid()
    plt.title("Média dos Melhores por Geração")
    plt.show()


if __name__ == "__main__":
    main()
