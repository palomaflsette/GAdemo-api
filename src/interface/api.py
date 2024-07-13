import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from services.genetic_algorithm_service import GeneticAlgorithmService
from domain.execution_characteristics import ExecutionCharacteristics
from domain.crossover_type import CrossoverType



app = FastAPI()
ga_service = GeneticAlgorithmService()


class ExecutionCharacteristicsModel(BaseModel):
    num_generations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    maximize: bool
    interval: List[int]
    crossover_type: dict


class CrossoverTypeModel(BaseModel):
    one_point: bool
    two_point: bool
    uniform: bool


@app.post("/run-experiments")
def run_experiments(func_str: str, exec_chars: ExecutionCharacteristicsModel, num_experiments: int):
    crossover_type = CrossoverType(**exec_chars.crossover_type)
    exec_chars_obj = ExecutionCharacteristics(
        exec_chars.num_generations,
        exec_chars.population_size,
        exec_chars.crossover_rate,
        exec_chars.mutation_rate,
        exec_chars.maximize,
        tuple(exec_chars.interval),
        crossover_type
    )
    best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation = ga_service.run_experiments(
        func_str, exec_chars_obj, crossover_type, num_experiments
    )
    return {
        "best_experiment_values": best_experiment_values,
        "best_individuals_per_generation": best_individuals_per_generation,
        "mean_best_individuals_per_generation": mean_best_individuals_per_generation.tolist()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
