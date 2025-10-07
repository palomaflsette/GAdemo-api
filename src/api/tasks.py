import os
import sys
from celery import Celery

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from application.ga_application_service import GeneticApplicationService
from domain.execution_parameters import ExecutionParameters

# Configura o Celery para usar a URL do Redis que o Heroku providenciou
celery_app = Celery(
    'tasks',
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

@celery_app.task
def run_ga_task(func_str, params_dict, num_experiments):
    """
    Esta função executa o trabalho pesado em segundo plano.
    """
    params = ExecutionParameters(**params_dict)
    ga_service = GeneticApplicationService()

    results = ga_service.executor.run_multiple_experiments(func_str, params, num_experiments)

    return results