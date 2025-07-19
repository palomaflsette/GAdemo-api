import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
import time
import logging
import uvicorn
import datetime
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware

from application.ga_application_service import GeneticApplicationService
from domain.execution_parameters import ExecutionParameters

# --- Configuração da Aplicação FastAPI ---
app = FastAPI(
    title="GADemo API",
    description="API para execução de experimentos com Algoritmos Genéticos.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    # Em produção, restrinja para o domínio do seu front-end
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint Raiz ---


@app.get("/")
def read_root():
    return {"message": "GADemo API está funcionando corretamente"}

# --- Endpoint Principal de Execução ---


@app.post("/run-experiments")
async def run_experiments(
    func_str: str = Query(...,
                          description="A função a ser otimizada em formato de string."),
    num_experiments: int = Query(
        ..., gt=0, description="O número de vezes que o experimento será executado."),
    # O corpo da requisição agora usa nosso modelo Pydantic unificado do domínio.
    # A validação é automática e robusta.
    params: ExecutionParameters = Body(...)
):
    """
    Executa um ou mais experimentos do Algoritmo Genético e retorna os resultados agregados.
    """
    start_time = time.time()

    # A função é pré-processada aqui na camada de interface
    func_str_safe = func_str.replace('^', '**')

    # 1. Instancia o serviço da camada de aplicação
    ga_service = GeneticApplicationService()

    # 2. Chama o método do serviço com uma assinatura muito mais limpa
    (
        best_experiment_values,
        best_individuals_per_generation,
        mean_best_individuals_per_generation,
        best_values_per_generation,
        last_generation_values,
    ) = await ga_service.run_experiments(func_str_safe, params, num_experiments)

    execution_time = time.time() - start_time

    # 3. Retorna o dicionário de resultados
    return {
        "best_experiment_values": best_experiment_values,
        "best_individuals_per_generation": best_individuals_per_generation,
        "mean_best_individuals_per_generation": mean_best_individuals_per_generation,
        "best_values_per_generation": best_values_per_generation,
        "last_generation_values": last_generation_values,
        "execution_time_seconds": round(execution_time, 4)
    }

# --- Bloco de Execução (para debug local) ---
if __name__ == "__main__":
    # Configuração de logging movida para dentro do bloco de execução
    log_file_path = os.path.join(os.path.dirname(__file__), "server_logs.log")
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format="%(asctime)s - %(message)s")

    logging.info(
        f"Starting GADemo API server at: {datetime.datetime.now():%H:%M:%S}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
