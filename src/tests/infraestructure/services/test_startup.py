import pytest
from unittest.mock import MagicMock
from app.infraestructure.genetic_algorithm_executor import GeneticAlgorithmExecutor
from app.domain.execution_characteristics import ExecutionCharacteristics
from app.domain.crossover_type import CrossoverType

@pytest.fixture
def exec_chars():
    # Criando uma instância de ExecutionCharacteristics com os parâmetros necessários
    return ExecutionCharacteristics(
        num_generations=5,
        population_size=5,
        crossover_rate=1.0,
        mutation_rate=1.0,
        maximize=True,
        interval=(0, 10),
        crossover_type=CrossoverType(one_point=True, two_point=False, uniform=False)
    )

@pytest.fixture
def mock_executor(mocker):
    # Mockando o método 'run_multiple_experiments' de GeneticAlgorithmExecutor
    mock_executor = mocker.patch('app.infraestructure.genetic_algorithm_executor.GeneticAlgorithmExecutor')
    return mock_executor

def test_main(mock_executor, exec_chars):
    # Mockando o retorno de 'run_multiple_experiments'
    mock_executor.return_value.run_multiple_experiments.return_value = (
        [10, 20, 30, 40],  # melhores valores de cada experimento
        [[1, 2, 3, 4] for _ in range(5)],  # melhores indivíduos por geração
        [10, 20, 30, 40, 50]  # média dos melhores indivíduos por geração
    )
    
    # Chama a função principal (main)
    from app.main import main  # Ajuste o caminho se necessário
    main()

    # Verifique se o método 'run_multiple_experiments' foi chamado corretamente
    mock_executor.return_value.run_multiple_experiments.assert_called_once_with(
        'x**2', exec_chars, exec_chars.crossover_type, num_experiments=4
    )

    # Verifique se o print foi chamado para os melhores valores e tabela (simulação de print)
    # Isso pode ser feito com patching nos prints, ou verificando que as listas retornadas são as esperadas.
    
    # Teste de resultados esperados
    result_best_values, result_best_individuals, result_mean_best_values = mock_executor.return_value.run_multiple_experiments()
    
    assert result_best_values == [10, 20, 30, 40]
    assert len(result_best_individuals) == 5  # 5 gerações
    assert result_mean_best_values == [10, 20, 30, 40, 50]
