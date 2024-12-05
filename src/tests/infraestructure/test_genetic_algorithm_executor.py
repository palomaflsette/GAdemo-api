import pytest
import sympy as sp
from unittest.mock import Mock
from app.infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor

def test_get_function_single_variable():
    ga_executor = GeneticAlgorithmExecutor()
    func_str = "x**2 + 3*x + 5"
    func, x, y = ga_executor.get_function(func_str)
    assert func == sp.sympify("x**2 + 3*x + 5 + (y**2 + 3*y + 5)")

def test_get_function_two_variables():
    ga_executor = GeneticAlgorithmExecutor()
    func_str = "x**2 + y**2"
    func, x, y = ga_executor.get_function(func_str)
    assert func == sp.sympify("x**2 + y**2")

def test_normalize_fitness():
    ga_executor = GeneticAlgorithmExecutor()
    fitnesses = [10, 20, 30, 40]
    normalized = ga_executor.normalize_fitness(fitnesses, 0, 1)
    assert normalized == [0.0, 0.3333333333333333, 0.6666666666666666, 1.0]

def test_normalize_fitness_equal_values():
    ga_executor = GeneticAlgorithmExecutor()
    fitnesses = [10, 10, 10]
    normalized = ga_executor.normalize_fitness(fitnesses, 0, 1)
    assert normalized == [0, 0, 0]

def test_evaluate_func():
    ga_executor = GeneticAlgorithmExecutor()

    # Teste para função de duas variáveis
    func = sp.sympify("x + y")
    x, y = sp.symbols("x y")
    individual = [3, 4]
    result = ga_executor.evaluate_func(individual, func, x, y)
    assert result == (7.0,), f"Expected (7.0,), but got {result}"

    # Teste para função de uma variável
    func = sp.sympify("x**2")
    x = sp.symbols("x")
    individual = [5]
    result = ga_executor.evaluate_func(individual, func, x)
    assert result == (25.0,), f"Expected (25.0,), but got {result}"

    # Teste com função constante
    func = sp.sympify("42")
    x, y = sp.symbols("x y")
    individual = [1, 2]
    result = ga_executor.evaluate_func(individual, func, x, y)
    assert result == (42.0,), f"Expected (42.0,), but got {result}"

    # Teste com função de expressão mais complexa
    func = sp.sympify("x**2 + y**2 + x*y")
    x, y = sp.symbols("x y")
    individual = [2, 3]
    result = ga_executor.evaluate_func(individual, func, x, y)
    assert result == (19.0,), f"Expected (19.0,), but got {result}"

    # Teste com valores negativos
    func = sp.sympify("x - y")
    x, y = sp.symbols("x y")
    individual = [-3, -7]
    result = ga_executor.evaluate_func(individual, func, x, y)
    assert result == (4.0,), f"Expected (4.0,), but got {result}"

    # Teste com ponto flutuante
    func = sp.sympify("x / 2 + y")
    x, y = sp.symbols("x y")
    individual = [5.5, 4.5]
    result = ga_executor.evaluate_func(individual, func, x, y)
    assert result == (7.25,), f"Expected (7.25,), but got {result}"


def test_mutate_within_bounds():
    ga_executor = GeneticAlgorithmExecutor()
    individual = [5, 15]
    interval = [0, 10]
    mutated_individual, = ga_executor.mutate_within_bounds(
        individual, interval, mu=0, sigma=1, indpb=1.0)
    assert all(interval[0] <= gene <= interval[1] for gene in mutated_individual)


def test_run_genetic_algorithm():
    ga_executor = GeneticAlgorithmExecutor()
    exec_chars = Mock(maximize=True, interval=[-10, 10], population_size=5, 
                      num_generations=2, normalize_linear=False, 
                      steady_state=False, crossover_rate=0.9, mutation_rate=0.1,
                      gap=0.5, steady_state_without_duplicateds=False,
                      elitism=False, normalize_min=0, normalize_max=1)
    
    cross_type = Mock(one_point=True, two_point=False, uniform=False)
    population, best_individuals, last_generation_values = ga_executor.run_genetic_algorithm(
        "x**2", exec_chars, cross_type)
    
    assert len(population) == exec_chars.population_size
    assert len(best_individuals) == exec_chars.num_generations
