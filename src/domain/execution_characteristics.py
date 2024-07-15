import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from domain.crossover_type import CrossoverType

class ExecutionCharacteristics:
    def __init__(self, num_generations: int, population_size: int, crossover_rate: float, mutation_rate: float, maximize: bool, interval: tuple, crossover_type: CrossoverType):
        self.num_generations = num_generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.maximize = maximize
        self.interval = interval
        self.crossover_type = crossover_type