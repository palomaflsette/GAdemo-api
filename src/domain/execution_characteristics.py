import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from domain.crossover_type import CrossoverType

class ExecutionCharacteristics:
    def __init__(self, num_generations: int, population_size: int, crossover_rate: float, 
                mutation_rate: float, normalize_linear, normalize_min, normalize_max, maximize: bool, interval: tuple, crossover_type: CrossoverType, elitism: bool,
                steady_state: bool, steady_state_without_duplicateds: bool , gap: float):
        self.num_generations = num_generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.normalize_linear = normalize_linear
        self.normalize_min = normalize_min
        self.normalize_max = normalize_max
        self.maximize = maximize
        self.interval = interval
        self.crossover_type = crossover_type
        self.elitism = elitism
        self.steady_state = steady_state
        self.steady_state_without_duplicateds = steady_state_without_duplicateds
        self.gap = gap

