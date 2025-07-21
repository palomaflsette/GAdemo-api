import sys
import os
import logging
import time
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import datetime
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, Body

from domain.execution_parameters import ExecutionParameters
from application.ga_application_service import GeneticApplicationService


app = FastAPI(
    title="GADemo API",
    description="API para execução de experimentos com Algoritmos Genéticos.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "GADemo API está funcionando corretamente"}


@app.post("/run-experiments")
async def run_experiments(
    func_str: str = Query(...,
                          description="A função a ser otimizada em formato de string."),
    num_experiments: int = Query(
        ..., gt=0, description="O número de vezes que o experimento será executado."),

    params: ExecutionParameters = Body(...)
):
    """
    Executa um ou mais experimentos do Algoritmo Genético e retorna os resultados agregados.
    
    """
    print(f"DEBUG: Parâmetros recebidos na API:")
    print(f"  - normalize_linear = {params.normalize_linear}")
    print(f"  - steady_state_removal = {params.steady_state_removal}")
    print(
        f"  - steady_state_with_duplicates = {params.steady_state_with_duplicates}")
    print(
        f"  - steady_state_without_duplicates = {params.steady_state_without_duplicates}")
    print(f"  - gap = {params.gap}")
    print(f"  - elitism = {params.elitism}")
    print(f"  - crossover = {params.crossover_type}")

    start_time = time.time()

    func_str_safe = func_str.replace('^', '**')

    ga_service = GeneticApplicationService()

    (
        best_experiment_values,
        best_individuals_per_generation,
        mean_best_individuals_per_generation,
        best_values_per_generation,
        last_generation_values,
    ) = await ga_service.run_experiments(func_str_safe, params, num_experiments)

    execution_time = time.time() - start_time

    return {
        "best_experiment_values": best_experiment_values,
        "best_individuals_per_generation": best_individuals_per_generation,
        "mean_best_individuals_per_generation": mean_best_individuals_per_generation,
        "best_values_per_generation": best_values_per_generation,
        "last_generation_values": last_generation_values,
        "execution_time_seconds": round(execution_time, 4),
        "parameters_used": {
            "steady_state_removal": params.steady_state_removal,
            "gap": params.gap,
            "steady_state_with_duplicates": params.steady_state_with_duplicates,
            "steady_state_without_duplicates": params.steady_state_without_duplicates
        }
    }

if __name__ == "__main__":
    log_file_path = os.path.join(os.path.dirname(__file__), "server_logs.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format="%(asctime)s - %(message)s")

    logging.info(
        f"Starting GADemo API server at: {datetime.datetime.now():%H:%M:%S}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
