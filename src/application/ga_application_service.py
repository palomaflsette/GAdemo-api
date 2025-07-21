import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


from domain.execution_parameters import CrossoverType, ExecutionParameters
from core.ga_executor import GeneticAlgorithmExecutor

class GeneticApplicationService:
    def __init__(self):
        self.executor = GeneticAlgorithmExecutor()

    async def run_experiments(self, func_str: str, params: ExecutionParameters, num_experiments: int):
        """
        Orquestra a execução de múltiplos experimentos de forma paralela.

        recebe ExecutionParameters completo ao invés de apenas CrossoverType.
        Isso permite que o executor acesse todos os parâmetros, incluindo o novo 
        'steady_state_removal'.

        Args:
            func_str: Função objetivo como string
            params: Parâmetros completos da execução (incluindo steady_state_removal)
            num_experiments: Número de experimentos a executar

        Returns:
            Tupla com resultados agregados de todos os experimentos
        """
        results = await self.executor.run_multiple_experiments(
            func_str,
            params, 
            num_experiments
        )

        return results
