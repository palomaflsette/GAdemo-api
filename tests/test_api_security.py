"""Testes de integração HTTP dos critérios de aceitação da Prioridade 1.

Requer httpx (via fastapi.testclient). Se não estiver instalado, os testes são
pulados — o parser já é coberto por test_safe_expression.py.
"""
import pytest

pytest.importorskip("httpx", reason="fastapi.testclient precisa de httpx")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)

# Corpo mínimo e rápido (poucas gerações/indivíduos) para os testes que rodam o GA.
PARAMS_BASE = {
    "num_generations": 3,
    "population_size": 10,
    "crossover_rate": 0.65,
    "mutation_rate": 0.05,
    "maximize": True,
    "interval": [-100, 100],
}


def _post(func_str, num_experiments=1, **param_overrides):
    params = {**PARAMS_BASE, **param_overrides}
    return client.post(
        "/run-experiments",
        params={"func_str": func_str, "num_experiments": num_experiments},
        json=params,
    )


def test_f6_e_aceita():
    resp = _post("0.5 - (sin(sqrt(x^2 + y^2))^2 - 0.5) / (1 + 0.001 * (x^2 + y^2))^2")
    assert resp.status_code == 200
    assert "best_experiment_values" in resp.json()


def test_funcao_simples_roda():
    resp = _post("sin(x) + cos(y)")
    assert resp.status_code == 200


def test_codigo_malicioso_retorna_400():
    resp = _post("__import__('math').factorial(5) + x")
    assert resp.status_code == 400


def test_acesso_a_atributo_retorna_400():
    resp = _post("x.__class__")
    assert resp.status_code == 400


def test_num_experiments_zero_e_rejeitado():
    assert _post("sin(x)", num_experiments=0).status_code == 422


def test_num_experiments_acima_do_teto_e_rejeitado():
    assert _post("sin(x)", num_experiments=999).status_code == 422


def test_population_acima_do_teto_e_rejeitada():
    assert _post("sin(x)", population_size=999999).status_code == 422


def test_interval_invertido_e_rejeitado():
    assert _post("sin(x)", interval=[100, -100]).status_code == 422
