"""Smoke test do pipeline completo nos municípios de SC (dataset fornecido por professor).

Roda ``experimentos/configs/municipios_sc.yaml`` (15 atributos numéricos
discretizados + 11 categóricos nativos, alvo Urbano/Rural 206/86) de ponta a ponta
e confere só propriedades estruturais: a execução termina, cada linha recebe uma
predição OOF, a acurácia fica abaixo de 100% (o dado tem sobreposição entre as
classes; um valor perfeito indicaria vazamento do alvo) e a classe minoritária é
predita alguma vez. Não confere números exatos nem exige ganho sobre a classe
majoritária: o adendo discute esse resultado.
"""

from __future__ import annotations

import pytest

from experimentos.runner import REPO_ROOT, load_config, run_experiment

CONFIG_PATH = REPO_ROOT / "experimentos" / "configs" / "municipios_sc.yaml"


@pytest.fixture(scope="module")
def municipios_run():
    config = load_config(CONFIG_PATH)
    assert config["discretizar"] is True
    assert 0 < len(config["continuous_columns"]) < len(config["feature_columns"])  # atributos mistos
    return run_experiment(config)


def test_every_row_gets_exactly_one_oof_prediction(municipios_run):
    oof = municipios_run["artifacts"]["oof"]
    assert sorted(r["row_id"] for r in oof) == list(range(len(oof)))
    assert len(oof) == municipios_run["summary"]["oof_metrics"]["n"] == 292
    assert {r["y_pred"] for r in oof} <= {"Urbano", "Rural"}


def test_target_split_and_no_perfect_accuracy(municipios_run):
    summary = municipios_run["summary"]
    assert summary["majority_baseline_metrics"]["accuracy"] == pytest.approx(206 / 292)
    assert summary["oof_metrics"]["accuracy"] < 1.0  # 100% indicaria vazamento do alvo


def test_rules_are_produced_and_minority_class_is_predicted(municipios_run):
    summary = municipios_run["summary"]
    assert summary["num_final_rules_mean"] > 0
    assert summary["oof_recall_positive_class"] > 0  # não degenera para a classe majoritária
