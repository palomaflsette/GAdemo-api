

import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, Query, Body
from pydantic import BaseModel
from typing import List
import datetime
import logging
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from services.genetic_algorithm_service import GeneticAlgorithmService

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "API funcionando corretamente"}

# Configurando o CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

log_file_path = os.path.join(os.path.dirname(__file__), "server_logs.log")
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format="%(asctime)s - %(message)s")


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

@app.post("/run-experiments")
async def run_experiments(
    func_str: str = Query(...),
    num_experiments: int = Query(...),
    exec_chars: ExecutionCharacteristicsModel = Body(...)
):
    func_str = func_str.replace('^', '**')

    ga_service = GeneticAlgorithmService()
    best_experiment_values, best_individuals_per_generation, mean_best_individuals_per_generation, best_values_per_generation, last_generation_values = await ga_service.run_experiments(
        func_str, exec_chars, exec_chars.crossover_type, num_experiments
    )

    return {
        "best_experiment_values": best_experiment_values,
        "best_individuals_per_generation": best_individuals_per_generation,
        "mean_best_individuals_per_generation": mean_best_individuals_per_generation,
        "best_values_per_generation": best_values_per_generation,
        "last_generation_values": last_generation_values
    }

if __name__ == "__main__":
    logging.info("Starting server at: %s",
                 datetime.datetime.now().strftime("%H:%M:%S"))
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, workers=40)
    except Exception as e:
        logging.error("Exception occurred: %s", e)
    finally:
        logging.info("Server stopped at: %s",
                     datetime.datetime.now().strftime("%H:%M:%S"))
