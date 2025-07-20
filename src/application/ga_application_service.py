import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from core.ga_executor import GeneticAlgorithmExecutor
from domain.execution_parameters import CrossoverType


class GeneticApplicationService:
    def __init__(self):
        self.executor = GeneticAlgorithmExecutor()

    async def run_experiments(self, func_str: str,  cross_type: CrossoverType, num_experiments: int):
        """
        Orquestra a execução de múltiplos experimentos de forma paralela.

        Este método delega a execução e a agregação de resultados diretamente
        para o executor, que já possui uma implementação otimizada para isso.
        """
        results = await self.executor.run_multiple_experiments(
            func_str,
            cross_type,
            num_experiments
        )

        return results
