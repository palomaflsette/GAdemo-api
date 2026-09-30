"""Testes do parser seguro de expressões (Prioridade 1: fechar o RCE).

Os casos de recusa da seção 2 do poc_seguro.py têm que ser recusados aqui.
"""
import math

import pytest

from core.ga_executor import (
    GeneticAlgorithmExecutor,
    UnsafeExpressionError,
    validate_expression,
)

# Função padrão do front (F6), já com ** no lugar de ^ (como chega à API).
F6 = "0.5 - (sin(sqrt(x**2 + y**2))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2"

# Funções que os botões do teclado geram — todas têm que passar.
FUNCOES_TECLADO = [
    "sin(x)",
    "cos(x)",
    "tan(x)",
    "log(x)",
    "sqrt(x)",
    "exp(x)",
    "pi",
]

EXPRESSOES_VALIDAS = [
    F6,
    "sin(x) + cos(y)",
    "x**2 + y**2",
    "-(x**2 + y**2)",
    "pi * x + y",
    "exp(-(x**2 + y**2))",
    *FUNCOES_TECLADO,
]

# Entradas que NÃO são matemática ou abusam de recursos — têm que ser recusadas.
EXPRESSOES_PERIGOSAS = [
    # --- seção 2 do poc_seguro.py ---
    "__import__('math').factorial(5) + x",
    "__import__('math').factorial(5)*0 + x**2",
    "x.__class__",
    # --- outros vetores ---
    "().__class__.__bases__",
    "os.system('ls')",
    "eval('1+1')",
    "open('/etc/passwd').read()",
    "x if x else y",          # IfExp
    "[i for i in range(9)]",  # comprehension
    "lambda: 1",              # Lambda
    "z",                      # nome não permitido
    "sin(x, base=2)",         # argumento nomeado
    # --- anti-DoS ---
    "9**9**9**9",             # torre de expoentes
    "10**9999999",            # constante grande demais
    "x**2 + " + "1+" * 200 + "1",  # string longa demais
]


@pytest.mark.parametrize("expr", EXPRESSOES_VALIDAS)
def test_expressoes_validas_passam(expr):
    # Não deve levantar exceção.
    validate_expression(expr)


@pytest.mark.parametrize("expr", EXPRESSOES_PERIGOSAS)
def test_expressoes_perigosas_sao_recusadas(expr):
    with pytest.raises(UnsafeExpressionError):
        validate_expression(expr)


def test_get_function_constroi_e_avalia_f6():
    """F6 é aceita e produz um número finito num ponto conhecido."""
    executor = GeneticAlgorithmExecutor()
    func, _x, _y = executor._get_function(F6)
    valor = float(func(0.0, 0.0))
    assert math.isfinite(valor)
    # Em (0,0), F6 = 0.5 - (0 - 0.5)/1 = 1.0
    assert valor == pytest.approx(1.0)


def test_get_function_recusa_codigo_malicioso():
    executor = GeneticAlgorithmExecutor()
    with pytest.raises(UnsafeExpressionError):
        executor._get_function("__import__('os').getcwd() + x")


def test_validacao_nao_tem_efeito_colateral(tmp_path):
    """A validação usa ast.parse, não executa a entrada: o arquivo não é criado."""
    alvo = tmp_path / "prova.txt"
    payload = f"open('{alvo}', 'w').write('x') + x"
    with pytest.raises(UnsafeExpressionError):
        validate_expression(payload)
    assert not alvo.exists()
