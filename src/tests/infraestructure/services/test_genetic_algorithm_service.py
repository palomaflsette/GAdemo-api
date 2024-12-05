import pytest
from unittest.mock import MagicMock
from app.domain.execution_characteristics import ExecutionCharacteristics
from app.domain.crossover_type import CrossoverType
from app.infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor
from app.services.genetic_algorithm_service import GeneticAlgorithmService

@pytest.fixture
def mock_executor(mocker):
    # Mock do GeneticAlgorithmExecutor
    mock_executor = mocker.patch('app.infraestructure.genetic_algorithm_executor.GeneticAlgorithmExecutor')
    return mock_executor
    
@pytest.fixture
def exec_chars():
    # Cria uma instância de CrossoverType com o valor desejado para "uniform"
    crossover_type = CrossoverType(one_point=False, two_point=False, uniform=True)
    
    # Cria o objeto ExecutionCharacteristics com todos os parâmetros necessários
    return ExecutionCharacteristics(
        num_generations=5,
        population_size=5,
        crossover_rate=1.0,
        mutation_rate=1.0,
        normalize_linear=True,  # Adicionando este valor
        normalize_min=0,        # Adicionando este valor
        normalize_max=1,        # Adicionando este valor
        maximize=True,
        interval=(0, 10),
        crossover_type=CrossoverType(one_point=True, two_point=False, uniform=False),
        elitism=True,  # Adicionando este valor
        steady_state=True,  # Adicionando este valor
        steady_state_without_duplicateds=False,  # Adicionando este valor
        gap=0.05  # Adicionando este valor
    )

@pytest.fixture
def service(mock_executor):
    # Instancia o serviço com o mock do executor
    return GeneticAlgorithmService()

@pytest.mark.asyncio  # Marca o teste como assíncrono
async def test_run_experiments(service, mock_executor, exec_chars):
    # Configura o mock para retornar valores fictícios
    mock_executor.return_value.run_genetic_algorithm.return_value = (
        None,  # Retorno do primeiro valor (não usado)
        [(i, f'Individual {i}', i * 10) for i in range(10)],  # Exemplo de indivíduos (i, nome, valor)
        [i * 10 for i in range(10)]  # Valores da última geração
    )

    # Defina os parâmetros de entrada
    func_str = 'x**2'  # Altere para uma expressão matemática válida
    crossover_type = CrossoverType(one_point=False, two_point=False, uniform=True)
    num_experiments = 5

    # Chama o método assíncrono com await
    result = await service.run_experiments(func_str, exec_chars, crossover_type, num_experiments)

    # Desempacote o resultado
    best_experiment_values, best_individuals_per_experiment, mean_best_individuals_per_generation, best_values_per_generation, last_generation_values_per_experiment = result

    # Verifique se o resultado está conforme esperado
    assert len(best_experiment_values) == num_experiments
    assert len(best_individuals_per_experiment) == num_experiments
    assert len(mean_best_individuals_per_generation) == exec_chars.num_generations
    assert len(best_values_per_generation) == num_experiments
    assert len(last_generation_values_per_experiment) == num_experiments

    # Verifique a média dos melhores valores por geração
    assert result_mean_best_individuals_per_generation[0] == pytest.approx(45.0, rel=4.5e-01)


    # Verifique os valores dos melhores indivíduos por experimento
    for i in range(num_experiments):
        assert best_experiment_values[i] == 90  # A última posição do melhor indivíduo (i * 10)

    # Verifique a estrutura dos melhores indivíduos por experimento
    for i in range(num_experiments):
        assert len(best_individuals_per_experiment[i]) == 10

    # Verifique os valores da última geração por experimento
    for i in range(num_experiments):
        assert last_generation_values_per_experiment[i] == [i * 10 for i in range(10)]


from deap import creator, base

try:
    creator.create("Individual", base.Fitness, list)
except TypeError:
    pass  # A classe já existe
