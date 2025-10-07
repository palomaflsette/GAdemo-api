
import sys
import os
import logging
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datetime
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, Body
from celery.result import AsyncResult

from .tasks import celery_app, run_ga_task
from domain.execution_parameters import ExecutionParameters

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


@app.post("/run-experiments", status_code=202)  
async def post_experiment(
    func_str: str = Query(..., description="A função a ser otimizada em formato de string."),
    num_experiments: int = Query(..., gt=0, description="O número de vezes que o experimento será executado."),
    params: ExecutionParameters = Body(...)
):
    """
    Este endpoint agora recebe os parâmetros, despacha a tarefa para o worker
    e retorna IMEDIATAMENTE.
    """
    func_str_safe = func_str.replace('^', '**')
    
    params_dict = params.model_dump()

    task = run_ga_task.delay(func_str_safe, params_dict, num_experiments)
    
    return {"message": "Experimento recebido e está sendo processado.", "task_id": task.id}


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """
    NOVO ENDPOINT: O frontend usará este endpoint para perguntar 'Já terminou?'.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    
    if not task_result.ready():
        return {"status": "PENDING"}

    if task_result.failed():
        return {"status": "FAILED", "error_message": str(task_result.info)}

    results = task_result.get()
    
    (
        best_experiment_values,
        best_individuals_per_generation,
        mean_best_individuals_per_generation,
        best_values_per_generation,
        last_generation_values,
    ) = results
    
    return {
        "status": "COMPLETED",
        "data": {
            "best_experiment_values": best_experiment_values,
            "best_individuals_per_generation": best_individuals_per_generation,
            "mean_best_individuals_per_generation": mean_best_individuals_per_generation,
            "best_values_per_generation": best_values_per_generation,
            "last_generation_values": last_generation_values,
        }
    }


if __name__ == "__main__":
    log_file_path = os.path.join(os.path.dirname(__file__), "server_logs.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format="%(asctime)s - %(message)s")

    logging.info(
        f"Starting GADemo API server at: {datetime.datetime.now():%H:%M:%S}")
    uvicorn.run(app, host="0.0.0.0", port=8000)