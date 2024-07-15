import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Query, Body

from services.genetic_algorithm_service import GeneticAlgorithmService


app = FastAPI()


class CrossoverTypeModel(BaseModel):
    one_point: bool = True
    two_point: bool = False
    uniform: bool = False


class ExecutionCharacteristicsModel(BaseModel):
    num_generations: int
    population_size: int
    crossover_rate: float
    mutation_rate: float
    maximize: bool
    interval: List[int]
    crossover_type: CrossoverTypeModel
    normalize_linear: bool = False
    normalize_min: int = 0
    normalize_max: int = 100
    elitism: bool = False
    steady_state: bool = False
    steady_state_without_duplicateds: bool = False
    gap: float = 0.0

    class Config:
        schema_extra = {
            "example": {
                "num_generations": 0,
                "population_size": 0,
                "crossover_rate": 0,
                "mutation_rate": 0,
                "maximize": True,
                "interval": [0],
                "crossover_type": {
                    "one_point": True,
                    "two_point": False,
                    "uniform": False
                }
            }
        }
        

@app.post("/run-experiments")
def run_experiments(
    func_str: str = Query(...),
    num_experiments: int = Query(...),
    exec_chars: ExecutionCharacteristicsModel = Body(...)
):
    ga_service = GeneticAlgorithmService()
    best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation = ga_service.run_experiments(
        func_str, exec_chars, exec_chars.crossover_type, num_experiments
    )
    return {
        "best_experiment_values": best_experiment_values,
        "best_individuals_per_generation": best_individuals_per_generation,
        "mean_best_individuals_per_generation": mean_best_individuals_per_generation
    }
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
