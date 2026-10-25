"""Smoke test do pipeline completo com atributos categóricos nativos reais.

Roda ``experimentos/configs/mushroom.yaml`` (UCI Mushroom: 22 atributos
categóricos, sem discretização) de ponta a ponta e confere só propriedades
estruturais — não números exatos: a execução termina, cada linha recebe uma
predição OOF e a acurácia fica acima da baseline da classe majoritária.
Complementa os testes sintéticos de ``test_runner.py`` (achado A08).
"""

from __future__ import annotations

import pytest

from experimentos.runner import REPO_ROOT, load_config, run_experiment

CONFIG_PATH = REPO_ROOT / "experimentos" / "configs" / "mushroom.yaml"


@pytest.fixture(scope="module")
def mushroom_run():
    config = load_config(CONFIG_PATH)
    assert config["discretizar"] is False
    assert not config["continuous_columns"]
    return run_experiment(config)


def test_every_row_gets_exactly_one_oof_prediction(mushroom_run):
    oof = mushroom_run["artifacts"]["oof"]
    assert sorted(r["row_id"] for r in oof) == list(range(len(oof)))
    assert len(oof) == mushroom_run["summary"]["oof_metrics"]["n"] > 8000
    assert {r["y_pred"] for r in oof} <= {"e", "p"}


def test_accuracy_beats_majority_class_baseline(mushroom_run):
    summary = mushroom_run["summary"]
    accuracy = summary["oof_metrics"]["accuracy"]
    baseline = summary["majority_baseline_metrics"]["accuracy"]
    assert 0.5 < baseline < 0.55  # classe majoritária (comestível) ~51,8%
    assert accuracy > baseline


def test_items_stay_categorical_and_rules_are_produced(mushroom_run):
    assert mushroom_run["summary"]["num_final_rules_mean"] > 0
    assert mushroom_run["summary"]["discretizar"] is False
