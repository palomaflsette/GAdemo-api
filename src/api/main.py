import sys
import os
import logging
import time
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import datetime
import uvicorn

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, Body, HTTPException

from domain.execution_parameters import ExecutionParameters
from application.ga_application_service import GeneticApplicationService
from core.ga_executor import validate_expression, UnsafeExpressionError


app = FastAPI(
    title="GADemo API",
    description="API para execução de experimentos com Algoritmos Genéticos.",
    version="1.0.0"
)

# Origens liberadas no CORS. Por padrão, os domínios reais do front e o dev
# local; o deploy institucional (Maxwell/VRAC) pode sobrescrever pela variável
# de ambiente GADEMO_ALLOWED_ORIGINS (lista separada por vírgula) sem mexer no
# código. allow_credentials fica False porque o front não envia credenciais —
# e ["*"] com credentials=True seria, aliás, uma combinação inválida.
_DEFAULT_ALLOWED_ORIGINS = [
    "https://palomasette.com",
    "https://www.palomasette.com",
    "https://palomaflsette.github.io",
]
_env_origins = os.getenv("GADEMO_ALLOWED_ORIGINS", "").strip()
allowed_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _DEFAULT_ALLOWED_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # localhost/127.0.0.1 em qualquer porta, para desenvolvimento.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
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
        ..., gt=0, le=50, description="O número de vezes que o experimento será executado (1 a 50)."),

    params: ExecutionParameters = Body(...)
):
    """
    Executa um ou mais experimentos do Algoritmo Genético e retorna os resultados agregados.
    
    """
    start_time = time.time()

    func_str_safe = func_str.replace('^', '**')

    # Valida a expressão ANTES de executar qualquer coisa: entrada que não seja
    # função matemática é recusada com 400, sem efeito colateral.
    try:
        validate_expression(func_str_safe)
    except UnsafeExpressionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ga_service = GeneticApplicationService()

    try:
        (
            best_experiment_values,
            best_individuals_per_generation,
            mean_best_individuals_per_generation,
            best_values_per_generation,
            last_generation_values,
        ) = await ga_service.run_experiments(func_str_safe, params, num_experiments)
    except UnsafeExpressionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
