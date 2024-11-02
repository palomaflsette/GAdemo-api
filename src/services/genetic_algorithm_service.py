import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor
from domain.execution_characteristics import ExecutionCharacteristics
from domain.crossover_type import CrossoverType

class GeneticAlgorithmService:
    def __init__(self):
        self.executor = GeneticAlgorithmExecutor()

    async def run_experiments(self, func_str, exec_chars, cross_type, num_experiments):
        best_experiment_values = []
        best_individuals_per_experiment = []
        best_values_per_generation = []
        last_generation_values_per_experiment = []

        for _ in range(num_experiments):
            _, best_individuals, last_generation_values = self.executor.run_genetic_algorithm(
                func_str, exec_chars, cross_type
            )

            best_experiment_values.append(best_individuals[-1][2])
            best_individuals_per_experiment.append([ind[1] for ind in best_individuals])
            best_values_per_generation.append([ind[2] for ind in best_individuals])
            last_generation_values_per_experiment.append(last_generation_values)

        mean_best_individuals_per_generation = [
            float(np.mean([best_values_per_generation[exp][gen] for exp in range(num_experiments)]))
            for gen in range(exec_chars.num_generations)
        ]

        return best_experiment_values, best_individuals_per_experiment, mean_best_individuals_per_generation, best_values_per_generation, last_generation_values_per_experiment
