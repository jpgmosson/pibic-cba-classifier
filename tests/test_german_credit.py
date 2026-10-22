"""Smoke test do pipeline completo num dataset real difícil, de atributos mistos.

Roda ``experimentos/configs/german_credit.yaml`` (UCI Statlog German Credit:
7 atributos numéricos discretizados + 13 categóricos nativos, classe 70/30) de
ponta a ponta e confere só propriedades estruturais e uma faixa plausível —
não números exatos: a execução termina, cada linha recebe uma predição OOF e a
acurácia fica acima da baseline majoritária e abaixo de 90% (o dataset tem
sobreposição real entre as classes; um valor acima disso indicaria vazamento
ou outra mudança drástica de comportamento).
"""

from __future__ import annotations

import pytest

from experimentos.runner import REPO_ROOT, load_config, run_experiment

CONFIG_PATH = REPO_ROOT / "experimentos" / "configs" / "german_credit.yaml"


@pytest.fixture(scope="module")
def german_run():
    config = load_config(CONFIG_PATH)
    assert config["discretizar"] is True
    assert 0 < len(config["continuous_columns"]) < len(config["feature_columns"])  # atributos mistos
    return run_experiment(config)


def test_every_row_gets_exactly_one_oof_prediction(german_run):
    oof = german_run["artifacts"]["oof"]
    assert sorted(r["row_id"] for r in oof) == list(range(len(oof)))
    assert len(oof) == german_run["summary"]["oof_metrics"]["n"] == 1000
    assert {r["y_pred"] for r in oof} <= {"1", "2"}


def test_accuracy_is_above_majority_baseline_and_below_ceiling(german_run):
    summary = german_run["summary"]
    accuracy = summary["oof_metrics"]["accuracy"]
    baseline = summary["majority_baseline_metrics"]["accuracy"]
    assert baseline == pytest.approx(0.7)  # classe majoritária (bom pagador) = 70%
    assert baseline < accuracy < 0.9


def test_rules_are_produced_and_minority_class_is_predicted(german_run):
    summary = german_run["summary"]
    assert summary["num_final_rules_mean"] > 0
    assert summary["oof_recall_positive_class"] > 0  # não degenera para a classe majoritária
