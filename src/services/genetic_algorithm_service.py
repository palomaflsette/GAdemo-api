import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor
from domain.execution_characteristics import ExecutionCharacteristics
from domain.crossover_type import CrossoverType


class GeneticAlgorithmService:
    def __init__(self):
        self.executor = GeneticAlgorithmExecutor()

    def run_experiments(self, func_str: str, exec_chars: ExecutionCharacteristics, cross_type: CrossoverType, num_experiments: int):
        best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation = self.executor.run_multiple_experiments(
            func_str, exec_chars, cross_type, num_experiments
        )
        # Convertendo os resultados para tipos de dados padrão
        best_experiment_values = [float(val) for val in best_experiment_values]
        best_individuals_per_generation = [
            [[float(allele) for allele in ind] for ind in generation] for generation in best_individuals_per_generation]
        mean_best_individuals_per_generation = [
            [float(allele) for allele in ind] for ind in mean_best_individuals_per_generation]

        return best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation
