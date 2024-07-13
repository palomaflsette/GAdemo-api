import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from quarentine.function import Function
from src.domain.crossover_type import CrossoverType
from src.domain.execution_characteristics import ExecutionCharacteristics
from src.infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor


class GeneticAlgorithmService:
    def __init__(self):
        self.executor = GeneticAlgorithmExecutor()

    def run_experiments(self, func_str: str, exec_chars: ExecutionCharacteristics, cross_type: CrossoverType, num_experiments: int):
        func = Function(func_str)
        best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation = self.executor.run_multiple_experiments(
            func, exec_chars, cross_type, num_experiments
        )
        return best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation
