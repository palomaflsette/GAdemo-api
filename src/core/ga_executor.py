import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import sympy as sp
import concurrent.futures
from deap import base, creator, tools

from domain.execution_parameters import ExecutionParameters


def _calculate_elite_count(population_size: int) -> int:
    """Função auxiliar para determinar o número de elites."""
    return 2 if population_size % 2 == 0 else 1


class GeneticAlgorithmExecutor:
     """
     Motor de execução para o Algoritmo Genético.
     Esta classe contém toda a lógica de configuração e execução do DEAP.
     """

     def __init__(self):
          pass

 
     def _run_single_experiment(self, func_str: str, params: ExecutionParameters) -> tuple:
          """
          Executa uma única rodada do algoritmo genético.
          Esta versão garante que o valor logado no histórico seja sempre o fitness BRUTO,
          que é o valor informativo para análise de desempenho.
          """
          if "Fitness" in creator.__dict__:
               del creator.Fitness
          if "Individual" in creator.__dict__:
               del creator.Individual
          creator.create("Fitness", base.Fitness, weights=(
               1.0 if params.maximize else -1.0,))
          creator.create("Individual", list, fitness=creator.Fitness)
          func, x, y = self._get_function(func_str)
          toolbox = self._setup_toolbox(func, x, y, params)
          population = toolbox.population(n=params.population_size)
          fitnesses = list(map(toolbox.evaluate, population))
          for ind, fit in zip(population, fitnesses):
               ind.fitness.values = fit

          history = []

          # Loop principal das gerações
          for gen in range(params.num_generations):
               # A evolução acontece aqui. A população interna permanece com scores brutos.
               if params.steady_state_with_duplicates or params.steady_state_without_duplicates:
                    self._evolve_steady_state(population, toolbox, params)
               else:
                    self._evolve_generational(population, toolbox, params)

              # --- LÓGICA DE LOGGING ---
               # Pega o melhor indivíduo da geração
               best_individual = tools.selBest(population, 1)[0]
               # Pega o seu fitness BRUTO, que é o valor real e que queremos analisar
               best_raw_fitness = best_individual.fitness.values[0]
               # Adiciona o valor BRUTO e informativo ao histórico
               history.append(
                    (gen, best_individual[:], best_raw_fitness)
               )

          last_generation_values = [ind.fitness.values[0] for ind in population]
          return population, history, last_generation_values

     async def run_multiple_experiments(self, func_str: str, params: ExecutionParameters, num_experiments: int) -> tuple:
          """
          Ponto de entrada principal. Executa múltiplos experimentos em paralelo.
          """
          all_results = []
          with concurrent.futures.ThreadPoolExecutor() as executor:
               futures = [
                    executor.submit(self._run_single_experiment, func_str, params)
                    for _ in range(num_experiments)
               ]

               for future in concurrent.futures.as_completed(futures):
                    all_results.append(future.result())

          return self._aggregate_results(all_results, params)


     def _aggregate_results(self, all_results: list, params: ExecutionParameters) -> tuple:
         """Processa os resultados brutos de todas as execuções."""
         best_experiment_values = [max(ind[2] for ind in r[1])
                                   for r in all_results]
         best_individuals_per_experiment = [[ind[1]
                                             for ind in r[1]] for r in all_results]
         best_values_per_generation = [[ind[2]
                                        for ind in r[1]] for r in all_results]
         last_generation_values_per_experiment = [r[2] for r in all_results]

         mean_best_individuals_per_generation = [
             float(np.mean([exp_gen_values[gen]
                            for exp_gen_values in best_values_per_generation]))
             for gen in range(params.num_generations)
         ]

         return (
             best_experiment_values,
             best_individuals_per_experiment,
             mean_best_individuals_per_generation,
             best_values_per_generation,
             last_generation_values_per_experiment,
         )

     def _get_function(self, func_str: str):
          x, y = sp.symbols('x y')
          func_expr = sp.sympify(func_str)
          if len(func_expr.free_symbols) == 1 and x in func_expr.free_symbols:
               func_expr = func_expr + func_expr.subs(x, y)
          return sp.lambdify((x, y), func_expr, "numpy"), x, y

     def _setup_toolbox(self, func, x, y, params: ExecutionParameters) -> base.Toolbox:
          """Configura a toolbox do DEAP com os parâmetros da execução."""
          toolbox = base.Toolbox()
          interval = params.interval
          crossover_type = params.crossover_type

          toolbox.register("attr_float", random.uniform,
                              interval[0], interval[1])
          toolbox.register("individual", tools.initRepeat,
                              creator.Individual, toolbox.attr_float, n=2)
          toolbox.register("population", tools.initRepeat,
                              list, toolbox.individual)
          toolbox.register("evaluate", self._evaluate_func, func=func, x=x, y=y)

          if crossover_type.one_point:
               toolbox.register("mate", tools.cxOnePoint)
          elif crossover_type.two_point:
               toolbox.register("mate", tools.cxTwoPoint)
          elif crossover_type.uniform:
               toolbox.register("mate", tools.cxUniform, indpb=0.5)

          toolbox.register("mutate", self._mutate_within_bounds,
                              interval=interval, mu=0, sigma=10, indpb=0.7)
          toolbox.register("select", tools.selRoulette)
          return toolbox
     



     def _normalize_fitness(self, fitnesses: list, min_val: float, max_val: float) -> list:
          
          """Normaliza uma lista de valores de fitness para um novo intervalo [min_val, max_val]."""
          min_fit = min(fitnesses)
          max_fit = max(fitnesses)

          if min_fit == max_fit:
               return [min_val for _ in fitnesses]

          normalized_values = []
          for fit in fitnesses:
               norm_val = min_val + (max_val - min_val) * \
                    (fit - min_fit) / (max_fit - min_fit)

               final_val = max(min_val, min(norm_val, max_val))
               normalized_values.append(final_val)

          return normalized_values



     def _apply_crossover_mutation(self, offspring: list, toolbox: base.Toolbox, params: ExecutionParameters):
          """
          Aplica crossover e mutação de forma IDÊNTICA para Geracional e Steady-State.
          
          Este método garante que as operações genéticas sejam exatamente as mesmas,
          independentemente da estratégia evolutiva usada.
          
          Args:
               offspring: Lista de indivíduos para aplicar operações
               toolbox: Toolbox do DEAP com operadores configurados
               params: Parâmetros com taxas de crossover e mutação
          """
          # CROSSOVER - Exatamente igual para ambas estratégias
          for child1, child2 in zip(offspring[::2], offspring[1::2]):
               if random.random() < params.crossover_rate:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

          # MUTAÇÃO - Exatamente igual para ambas estratégias  
          for mutant in offspring:
               if random.random() < params.mutation_rate:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

          # AVALIAÇÃO - Exatamente igual para ambas estratégias
          invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
          fitnesses = map(toolbox.evaluate, invalid_ind)
          for ind, fit in zip(invalid_ind, fitnesses):
               ind.fitness.values = fit

     def _evolve_generational(self, population: list, toolbox: base.Toolbox, params: ExecutionParameters):
          """
          Evolução geracional - LIMPA, só cuida da lógica geracional.
          Crossover/mutação delegados para método unificado.
          """
          # ELITISMO
          elites = []
          if params.elitism:
               elite_count = _calculate_elite_count(params.population_size)
               elites = tools.selBest(population, elite_count)
               elites = list(map(toolbox.clone, elites))

          # SELEÇÃO (toda a população)
          if params.normalize_linear:
               selection_pool = list(map(toolbox.clone, population))
               raw_scores = [ind.fitness.values[0] for ind in selection_pool]
               normalized_scores = self._normalize_fitness(
                    raw_scores, params.normalize_min, params.normalize_max
               )
               for ind, norm_fit in zip(selection_pool, normalized_scores):
                    ind.fitness.values = (norm_fit,)
               offspring = toolbox.select(selection_pool, len(population))
          else:
               offspring = toolbox.select(population, len(population))
          
          offspring = list(map(toolbox.clone, offspring))

          # CROSSOVER + MUTAÇÃO (método unificado - GARANTIDAMENTE IGUAL)
          self._apply_crossover_mutation(offspring, toolbox, params)

          # NORMALIZAÇÃO PÓS-OPERAÇÃO (se necessário)
          if params.normalize_linear:
               for ind in offspring:
                    if ind.fitness.valid:
                         ind.fitness.values = toolbox.evaluate(ind)

          # SUBSTITUIÇÃO GERACIONAL (população inteira)
          population[:] = offspring

          # REINTRODUÇÃO DE ELITES
          if params.elitism and elites:
               population.extend(elites)
               population.sort(key=lambda ind: ind.fitness.values, reverse=params.maximize)
               population[:] = population[:params.population_size]

     def _evolve_steady_state(self, population: list, toolbox: base.Toolbox, params: ExecutionParameters):
          """
          Evolução Steady-State - CORRIGIDO para evitar erro de remove().
          Usa índices ao invés de referências de objetos.
          """
          # CÁLCULO DO GAP
          num_offspring = int(params.gap * len(population))
          if num_offspring == 0 and params.gap > 0:
               num_offspring = 1
          if num_offspring == 0:
               return 

          # SELEÇÃO DE PAIS (apenas fração da população)
          parents = tools.selRandom(population, k=num_offspring)
          offspring = list(map(toolbox.clone, parents))

          # CROSSOVER + MUTAÇÃO (método unificado)
          self._apply_crossover_mutation(offspring, toolbox, params)

          # CONTROLE DE DUPLICATAS
          if params.steady_state_without_duplicates:
               existing_individuals = {tuple(ind) for ind in population}
               unique_offspring = []
               for ind in offspring:
                    if tuple(ind) not in existing_individuals:
                         unique_offspring.append(ind)
                         existing_individuals.add(tuple(ind))
               offspring = unique_offspring

          if not offspring: 
               return 

          # ESTRATÉGIA DE REMOÇÃO
          removal_type = getattr(params, 'steady_state_removal', 'random')
          
          if removal_type == 'random':
               # ===== STEADY-STATE CLÁSSICO =====
               indices_to_remove = random.sample(range(len(population)), len(offspring))
               
          elif removal_type == 'worst':
               # ===== STEADY-STATE ELITISTA (CORRIGIDO) =====
               # Cria lista de (fitness, índice) e ordena
               fitness_index_pairs = [(ind.fitness.values[0], i) for i, ind in enumerate(population)]
               fitness_index_pairs.sort(key=lambda x: x[0], reverse=not params.maximize)  # Piores primeiro
               indices_to_remove = [pair[1] for pair in fitness_index_pairs[:len(offspring)]]
               
          elif removal_type == 'tournament':
               # ===== STEADY-STATE POR TORNEIO =====
               indices_to_remove = []
               available_indices = list(range(len(population)))
               
               for _ in range(len(offspring)):
                    # Torneio entre índices disponíveis
                    tournament_indices = random.sample(available_indices, min(2, len(available_indices)))
                    
                    # Seleciona o PIOR do torneio (menor fitness se maximizando)
                    if params.maximize:
                         worst_idx = min(tournament_indices, key=lambda i: population[i].fitness.values[0])
                    else:
                         worst_idx = max(tournament_indices, key=lambda i: population[i].fitness.values[0])
                         
                    indices_to_remove.append(worst_idx)
                    available_indices.remove(worst_idx) 
                    
          else:
               indices_to_remove = random.sample(range(len(population)), len(offspring))

          for idx in sorted(indices_to_remove, reverse=True):
               population.pop(idx)
          
          population.extend(offspring)

     def _evaluate_func(self, individual: list, func: callable, x: sp.Symbol, y: sp.Symbol) -> tuple:
          """
          Avalia o fitness de um indivíduo aplicando-o à função objetivo.
          Lida com funções de 1 ou 2 variáveis.
          """
          if y is not None and len(individual) == 2:
               x_val, y_val = individual
               result = func(x_val, y_val)
          else:
               x_val = individual[0]
               result = func(x_val, x_val) 

          return float(result),

     def _mutate_within_bounds(self, individual: list, interval: list, mu: float, sigma: float, indpb: float) -> tuple:
          """
          Aplica uma mutação gaussiana em cada gene do indivíduo, garantindo
          que os novos valores permaneçam dentro dos limites (interval).
          """
          for i in range(len(individual)):
               if random.random() < indpb:
                    # ruído gaussiano
                    individual[i] += random.gauss(mu, sigma)
                    individual[i] = max(interval[0], min(individual[i], interval[1]))
                    
          return individual,