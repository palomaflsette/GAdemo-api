"""Verificação rápida de uma instalação da API do GAdemo.

Rode com o Python do ambiente virtual que o serviço vai usar:

    caminho/do/venv/bin/python scripts/verificar_instalacao.py

Sem subir servidor e sem acessar a rede, confere três coisas:

1. a aplicação importa com as dependências instaladas;
2. uma função comum é otimizada de ponta a ponta (NumPy, SymPy e DEAP);
3. uma expressão proibida é recusada (a validação de segurança está ativa).

Termina com código 0 se tudo passar e 1 caso contrário. O scripts/atualizar.sh
usa esta verificação antes de trocar o ambiente do servidor.
"""
import asyncio
import os
import sys

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(RAIZ, "src"))


def main() -> int:
    try:
        from api.main import app
        from application.ga_application_service import GeneticApplicationService
        from core.ga_executor import UnsafeExpressionError, validate_expression
        from domain.execution_parameters import ExecutionParameters
    except Exception as exc:  # noqa: BLE001 - qualquer falha aqui reprova a instalação
        print(f"FALHA  importação da aplicação: {exc!r}")
        return 1
    print(f"ok     importação ({app.title} {app.version})")

    params = ExecutionParameters(
        num_generations=3,
        population_size=10,
        crossover_rate=0.65,
        mutation_rate=0.05,
        maximize=True,
        interval=[-100, 100],
    )
    try:
        resultado = asyncio.run(
            GeneticApplicationService().run_experiments("sin(x) + cos(y)", params, 1)
        )
        melhor = resultado[0][0]
    except Exception as exc:  # noqa: BLE001
        print(f"FALHA  experimento de teste: {exc!r}")
        return 1
    print(f"ok     experimento de teste (melhor valor encontrado: {melhor:.4f})")

    try:
        validate_expression("x.__class__")
    except UnsafeExpressionError:
        print("ok     expressão proibida recusada")
    else:
        print("FALHA  a expressão x.__class__ foi aceita")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
