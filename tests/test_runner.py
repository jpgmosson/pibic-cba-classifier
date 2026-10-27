"""Testes do runner: particionamento (A05/A06), contratos de configuração
(A09), ausência de vazamento da classe na inferência (A04), artefatos
auditáveis e métricas reproduzíveis a partir das predições OOF (A11/A12)."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter

import pandas as pd
import pytest

from experimentos import runner
from experimentos.metricas import compute_metrics, read_oof
from experimentos.runner import (
    SUMMARY_COLUMNS,
    load_config,
    migrate_summary_header,
    run_experiment,
    run_fold,
    stratified_group_kfold_indices,
    stratified_kfold_indices,
    validate_config,
    write_results,
)

# ------------------------------------------------------------- particionamento ---


def _check_partition(folds, n):
    flat = [i for fold in folds for i in fold]
    assert sorted(flat) == list(range(n))  # cada índice exatamente uma vez
    assert all(len(fold) > 0 for fold in folds)  # nenhuma dobra vazia


def test_stratified_kfold_rejects_the_empty_fold_case_of_the_report():
    with pytest.raises(ValueError):
        stratified_kfold_indices(["A", "B", "C"], k=3, seed=0)


@pytest.mark.parametrize("k", [0, 1, True, 2.5])
def test_stratified_kfold_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        stratified_kfold_indices(["A", "B"] * 5, k=k, seed=0)


def test_stratified_kfold_rejects_k_larger_than_smallest_class():
    with pytest.raises(ValueError, match="menor classe"):
        stratified_kfold_indices(["A"] * 20 + ["B"] * 3, k=5, seed=0)


def test_stratified_kfold_non_divisible_classes_are_balanced_and_disjoint():
    labels = ["A"] * 11 + ["B"] * 7 + ["C"] * 5  # nenhuma divisível por 5
    folds = stratified_kfold_indices(labels, k=5, seed=1)
    _check_partition(folds, len(labels))
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1
    for cls in "ABC":
        per_fold = [sum(labels[i] == cls for i in f) for f in folds]
        assert max(per_fold) - min(per_fold) <= 1


def test_stratified_kfold_is_reproducible_by_seed():
    labels = ["A"] * 10 + ["B"] * 10
    assert stratified_kfold_indices(labels, 5, 3) == stratified_kfold_indices(labels, 5, 3)


def _grouped_data(seed=0, n_groups=60):
    rng = random.Random(seed)
    labels, groups = [], []
    for g in range(n_groups):
        size = rng.choice([1, 1, 1, 2, 3])
        label = "pos" if g % 8 == 0 else "neg"
        labels += [label] * size
        groups += [f"g{g}"] * size
    return labels, groups


def test_group_kfold_has_empty_group_intersection_and_covers_every_row_once():
    labels, groups = _grouped_data()
    folds = stratified_group_kfold_indices(labels, groups, k=5, seed=4)
    _check_partition(folds, len(labels))
    for f, test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_groups = {groups[i] for i in range(len(labels)) if i not in test_set}
        assert not train_groups & {groups[i] for i in test_idx}, f"grupo compartilhado na dobra {f}"


def test_group_kfold_keeps_class_distribution_roughly_balanced():
    labels, groups = _grouped_data(n_groups=200)
    folds = stratified_group_kfold_indices(labels, groups, k=5, seed=4)
    pos = [sum(labels[i] == "pos" for i in f) for f in folds]
    assert max(pos) - min(pos) <= 3 and min(pos) > 0


def test_group_kfold_rejects_more_folds_than_groups_and_length_mismatch():
    with pytest.raises(ValueError, match="grupos"):
        stratified_group_kfold_indices(["a", "b", "a"], ["g1", "g1", "g2"], k=3, seed=0)
    with pytest.raises(ValueError):
        stratified_group_kfold_indices(["a", "b"], ["g1"], k=2, seed=0)


# ------------------------------------------------------------------ configuração ---

BASE_CONFIG = dict(
    dataset_path="x.csv",
    class_column="y",
    continuous_columns=["a"],
    n_bins=3,
    min_support=0.1,
    min_confidence=0.5,
    k_folds=3,
    seed=1,
    discretizar=True,
    csv_sep=",",
    feature_columns=None,
    positive_class=None,
    id_column=None,
    group_column=None,
    max_antecedent_length=None,
    cb_variant="m1",
    cba_qs_weights=None,
)


def cfg(**overrides):
    return validate_config({**BASE_CONFIG, **overrides})


def test_positive_class_yaml_boolean_matches_the_string_label():
    assert cfg(positive_class=True)["positive_class"] == "True"
    assert cfg(positive_class="True")["positive_class"] == "True"
    assert cfg(positive_class=1)["positive_class"] == "1"


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_confidence": float("nan")},
        {"min_confidence": -0.1},
        {"min_support": 0},
        {"min_support": float("inf")},
        {"n_bins": True},
        {"n_bins": 1},
        {"k_folds": 1},
        {"k_folds": 2.0},
        {"max_antecedent_length": 0},
        {"feature_columns": ["a", "y"]},
        {"feature_columns": ["a", "a"]},
        {"continuous_columns": ["y"]},
        {"feature_columns": ["a"], "continuous_columns": ["b"]},
        {"feature_columns": ["a", "id"], "id_column": "id"},
        {"feature_columns": ["a", "g"], "group_column": "g"},
        {"cb_variant": "outro"},
    ],
)
def test_validate_config_rejects_invalid_values_early(overrides):
    with pytest.raises(ValueError):
        cfg(**overrides)


def test_load_config_reads_repo_yaml_files():
    pokemon = load_config(runner.REPO_ROOT / "experimentos" / "configs" / "pokemon.yaml")
    assert pokemon["positive_class"] == "True" and pokemon["group_column"] == "#"
    iris = load_config(runner.REPO_ROOT / "experimentos" / "configs" / "iris.yaml")
    assert iris["group_column"] is None and iris["cb_variant"] == "m1"


# ------------------------------------------------- pipeline pequeno ponta a ponta ---


def make_dataset(tmp_path, n_groups=40, categorical_only=False):
    rng = random.Random(11)
    rows = []
    for g in range(n_groups):
        for _ in range(rng.choice([1, 1, 2, 3])):
            positive = g % 4 == 0
            a = rng.gauss(10 if positive else 0, 2)
            rows.append(
                {
                    "sid": g,
                    "name": f"s{g}_{len(rows)}",
                    "a": round(a, 3),
                    "color": rng.choice("rgb") if not positive else rng.choice("rr g".replace(" ", "")),
                    "y": "pos" if positive else "neg",
                }
            )
    df = pd.DataFrame(rows)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path, df


def small_config(path, **overrides):
    return validate_config(
        {
            **BASE_CONFIG,
            "dataset_path": str(path),
            "feature_columns": ["a", "color"],
            "continuous_columns": ["a"],
            "id_column": "name",
            "group_column": "sid",
            "positive_class": "pos",
            "k_folds": 4,
            "min_support": 0.05,
            **overrides,
        }
    )


def test_run_experiment_by_group_never_shares_a_group_and_scores_every_row_once(tmp_path):
    path, df = make_dataset(tmp_path)
    run = run_experiment(small_config(path))
    parts = run["artifacts"]["partitions"]
    assert all(p["groups_shared_train_test"] == 0 for p in parts)
    oof = run["artifacts"]["oof"]
    assert sorted(r["row_id"] for r in oof) == list(range(len(df)))
    assert run["summary"]["split"] == "group"


def test_row_split_can_share_groups_which_is_the_leak_the_group_split_removes(tmp_path):
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path, split="row"))
    assert any(p["groups_shared_train_test"] > 0 for p in run["artifacts"]["partitions"])


def test_group_column_and_id_are_not_used_as_predictors(tmp_path):
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path, feature_columns=None))
    rules = [r for fold in run["artifacts"]["rules"].values() for r in fold]
    used = {cond.split("=")[0] for r in rules for cond in r["antecedent"]}
    assert used <= {"a", "color"}


def test_unknown_positive_class_is_rejected_before_training(tmp_path):
    path, _ = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="positive_class"):
        run_experiment(small_config(path, positive_class="nao_existe"))


def test_boolean_target_and_yaml_true_give_a_defined_recall(tmp_path):
    rows = [{"v": float(i % 10), "flag": i % 5 == 0} for i in range(100)]
    path = tmp_path / "bool.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    config = validate_config(
        {
            **BASE_CONFIG,
            "dataset_path": str(path),
            "class_column": "flag",
            "continuous_columns": ["v"],
            "positive_class": True,  # booleano do YAML
            "min_support": 0.05,
            "k_folds": 5,
        }
    )
    run = run_experiment(config)
    assert run["summary"]["recall_positive_class_mean"] is not None
    assert run["summary"]["oof_recall_positive_class"] is not None


def test_categorical_only_dataset_runs_with_discretizar_true(tmp_path):
    path, _ = make_dataset(tmp_path)
    config = small_config(path, feature_columns=["color"], continuous_columns=[], discretizar=True)
    run = run_experiment(config)
    assert run["summary"]["oof_metrics"]["n"] > 0


def test_test_labels_do_not_influence_predictions(tmp_path):
    path, df = make_dataset(tmp_path)
    config = small_config(path)
    cut = 2 * len(df) // 3
    train = df.iloc[:cut][["a", "color", "y"]].reset_index(drop=True)
    test = df.iloc[cut:][["a", "color", "y"]].reset_index(drop=True)
    shuffled = test.copy()
    shuffled["y"] = random.Random(5).sample(list(test["y"]), len(test))
    flipped = test.copy()
    flipped["y"] = flipped["y"].map({"pos": "neg", "neg": "pos"})

    base = run_fold(train, test, config)["_predictions"]
    assert run_fold(train, shuffled, config)["_predictions"] == base
    assert run_fold(train, flipped, config)["_predictions"] == base


# ---------------------------------------------------- artefatos e reprodutibilidade ---


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    target = tmp_path / "resultados"
    monkeypatch.setattr(runner, "RESULTS_DIR", target)
    return target


def test_write_results_creates_auditable_artifacts_and_oof_metrics_are_reproducible(tmp_path, results_dir):
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path))
    detail = write_results(run, "data")
    run_dir = detail.parent
    for name in (
        "summary.json", "config.json", "environment.txt", "partitions.json", "discretization.json",
        "rules.json", "m1_history.json", "oof_predictions.csv", "metrics.json",
    ):
        assert (run_dir / name).exists(), name

    config_json = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_json["seed"] == 1 and len(config_json["dataset_sha256"]) == 64

    # métricas recalculadas só a partir das predições OOF salvas == resumo
    y_true, y_pred = read_oof(run_dir / "oof_predictions.csv")
    recomputed = compute_metrics(y_true, y_pred, labels=["neg", "pos"])
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mcc"):
        assert recomputed[key] == pytest.approx(summary["oof_metrics"][key])
    assert recomputed["confusion_matrix"]["matrix"] == summary["oof_metrics"]["confusion_matrix"]["matrix"]

    # histórico do M1: o prefixo escolhido é o primeiro de menor erro total
    history = json.loads((run_dir / "m1_history.json").read_text(encoding="utf-8"))
    for fold in history.values():
        totals = [p["total_errors"] for p in fold["prefixes"]]
        if totals:
            assert fold["chosen_prefix_length"] == totals.index(min(totals)) + 1


def test_two_runs_never_overwrite_each_other(tmp_path, results_dir):
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path))
    first = write_results(run, "data").parent.name
    second = write_results(run_experiment(small_config(path)), "data").parent.name
    assert first != second
    assert len({p.name for p in results_dir.iterdir() if p.is_dir()}) == 2
    with open(results_dir / "resumo.csv", encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2


def test_summary_csv_header_is_migrated_without_shifting_columns(results_dir):
    results_dir.mkdir()
    old_header = SUMMARY_COLUMNS[:15]
    path = results_dir / "resumo.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(old_header)
        writer.writerow([f"v{i}" for i in range(15)])
        writer.writerow([f"w{i}" for i in range(10)])  # linha ainda mais antiga (curta)
    migrate_summary_header(path)
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == SUMMARY_COLUMNS
    assert rows[0]["cb_variant"] == "v14" and rows[0]["run_id"] == ""
    assert rows[1]["total_elapsed_seconds"] == "w9" and rows[1]["cb_variant"] == ""


def test_incompatible_summary_header_is_refused(results_dir):
    results_dir.mkdir()
    (results_dir / "resumo.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        migrate_summary_header(results_dir / "resumo.csv")


# ----------------------------------------------------------------------- métricas ---


def test_compute_metrics_known_binary_example():
    y_true = ["p"] * 4 + ["n"] * 6
    y_pred = ["p", "p", "p", "n"] + ["n", "n", "n", "n", "p", "p"]
    m = compute_metrics(y_true, y_pred, labels=["n", "p"])
    # tp=3 fn=1 fp=2 tn=4
    assert m["confusion_matrix"]["matrix"] == [[4, 2], [1, 3]]
    assert m["accuracy"] == pytest.approx(0.7)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert m["per_class"]["p"]["precision"] == pytest.approx(3 / 5)
    assert m["per_class"]["p"]["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    expected_mcc = (3 * 4 - 2 * 1) / ((3 + 2) * (3 + 1) * (4 + 2) * (4 + 1)) ** 0.5
    assert m["mcc"] == pytest.approx(expected_mcc)


def test_majority_only_predictor_has_zero_mcc_and_half_balanced_accuracy():
    m = compute_metrics(["a"] * 9 + ["b"], ["a"] * 10, labels=["a", "b"])
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["mcc"] == 0.0 and m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["recall"] == 0.0


def test_compute_metrics_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        compute_metrics(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        compute_metrics([], [])


def test_frozen_cba_qs_variant_still_executes_through_the_runner(tmp_path):
    # Fumaça apenas: CBA-QS está congelado nesta baseline e não é validado aqui.
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path, cb_variant="cba_qs"))
    assert run["summary"]["cb_variant"] == "cba_qs" and run["summary"]["oof_metrics"]["n"] > 0


def test_uncertainty_separates_between_fold_dispersion_from_bootstrap_ci(tmp_path):
    path, df = make_dataset(tmp_path)
    run = run_experiment(small_config(path))
    unc = run["summary"]["oof_uncertainty"]
    assert unc["bootstrap"]["unit"] == "group"
    acc = run["summary"]["oof_metrics"]["accuracy"]
    lo, hi = unc["bootstrap"]["ci"]["accuracy"]
    assert lo <= acc <= hi and hi - lo > 0
    assert unc["between_folds"]["accuracy"]["min"] <= unc["between_folds"]["accuracy"]["max"]
    # reprodutível com a mesma seed
    assert run_experiment(small_config(path))["summary"]["oof_uncertainty"] == unc


def test_bootstrap_resamples_whole_groups():
    from experimentos.metricas import bootstrap_ci

    y = ["a", "a", "b", "b"]
    # 2 grupos: g1 acertado, g2 errado; toda reamostra tem acurácia em {0, 0.5, 1}
    ci = bootstrap_ci(y, ["a", "a", "a", "a"], ["g1", "g1", "g2", "g2"], ["a", "b"], reps=200, seed=1)
    assert ci["accuracy"][0] == 0.0 and ci["accuracy"][1] == 1.0


# ------------------------------------------- discretização MDL e poda pessimista no pipeline ---


@pytest.mark.parametrize(
    "overrides",
    [
        {"discretization_method": "outro"},
        {"prune": "sim"},
        {"prune": 1},
        {"prune_cf": 0},
        {"prune_cf": 1},
        {"prune_cf": float("nan")},
        {"prune_cf": True},
    ],
)
def test_validate_config_rejects_invalid_mdl_and_prune_options(overrides):
    with pytest.raises(ValueError):
        cfg(**overrides)


def test_new_options_default_to_the_baseline_behaviour():
    c = cfg()
    assert c["discretization_method"] == "quantile" and c["prune"] is False and c["prune_cf"] == 0.25


def test_n_bins_is_optional_when_not_using_quantiles():
    config = {k: v for k, v in BASE_CONFIG.items() if k != "n_bins"}
    assert validate_config(config)["n_bins"] == 3


def test_pipeline_runs_with_mdl_and_with_pruning(tmp_path):
    path, df = make_dataset(tmp_path)
    run = run_experiment(small_config(path, discretization_method="mdl"))
    assert run["summary"]["discretization_method"] == "mdl" and run["summary"]["prune"] is False
    disc = run["artifacts"]["discretization"][0]
    assert disc["a"]["method"].startswith("entropy-mdl") and disc["a"]["n_effective_bins"] >= 2  # 'a' separa as classes
    assert sorted(r["row_id"] for r in run["artifacts"]["oof"]) == list(range(len(df)))
    pruned = run_experiment(small_config(path, prune=True))
    plain = run_experiment(small_config(path))
    assert pruned["summary"]["prune"] is True
    assert pruned["summary"]["num_candidate_rules_mean"] <= plain["summary"]["num_candidate_rules_mean"]


def test_mdl_cuts_use_only_training_labels_and_test_labels_never_matter(tmp_path):
    path, df = make_dataset(tmp_path)
    config = small_config(path, discretization_method="mdl")
    cut = 2 * len(df) // 3
    train = df.iloc[:cut][["a", "color", "y"]].reset_index(drop=True)
    test = df.iloc[cut:][["a", "color", "y"]].reset_index(drop=True)
    flipped = test.assign(y=test["y"].map({"pos": "neg", "neg": "pos"}))
    base = run_fold(train, test, config)
    other = run_fold(train, flipped, config)
    assert other["_predictions"] == base["_predictions"]
    assert other["_discretization"] == base["_discretization"]  # cortes iguais: o teste não entra no fit


def test_runs_are_not_mixed_up_in_the_summary_csv(tmp_path, results_dir):
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path, discretization_method="mdl", prune=True))
    write_results(run, "data")
    with open(results_dir / "resumo.csv", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["discretization_method"] == "mdl" and row["prune"] == "True"
    assert "mdl_poda" in row["run_id"]


def test_iris_paper_config_reproduces_the_order_of_magnitude_of_the_published_cba(results_dir):
    """Regressão do experimento de fidelidade: Iris nas condições do artigo (publicado: erro 7,1%,
    72 CARs / 23 prCARs, 5 regras). Faixas largas, só para detectar regressões grosseiras."""
    config = load_config(runner.REPO_ROOT / "experimentos" / "configs" / "iris_paper.yaml")
    assert config["discretization_method"] == "mdl" and config["k_folds"] == 10 and config["max_antecedent_length"] is None
    plain = run_experiment(config)["summary"]
    pruned = run_experiment({**config, "prune": True})["summary"]
    assert 0.03 <= 1 - plain["oof_metrics"]["accuracy"] <= 0.10
    assert 3 <= plain["num_final_rules_mean"] <= 7
    assert 60 <= plain["num_candidate_rules_mean"] <= 130
    assert pruned["num_candidate_rules_mean"] < 0.5 * plain["num_candidate_rules_mean"]  # a poda corta a maioria das CARs


def test_command_line_flags_select_mdl_and_pruning(results_dir):
    runner.main(["--config", str(runner.REPO_ROOT / "experimentos" / "configs" / "iris.yaml"), "--discretization", "mdl", "--prune"])
    with open(results_dir / "resumo.csv", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["discretization_method"] == "mdl" and row["prune"] == "True" and "mdl_poda" in row["run_id"]


# ------------------------------------------------------------ mostrar_matriz ---


def test_mostrar_matriz_prints_the_saved_numbers_without_rerunning(tmp_path, results_dir):
    from io import StringIO

    from rich.console import Console

    from experimentos.mostrar_matriz import render

    path, _ = make_dataset(tmp_path)
    run_dir = write_results(run_experiment(small_config(path)), "data").parent
    matrix = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["oof"]["confusion_matrix"]["matrix"]

    out = StringIO()
    render(run_dir, Console(file=out, width=100, color_system=None))
    text = out.getvalue()

    assert "Matriz de confusão" in text and "data" in text
    assert "✔ matriz confere" in text
    for count in (c for row in matrix for c in row):
        assert str(count) in text


# ----------------------------------------------------------------- --show-rules ---


def test_show_rules_only_adds_text_and_never_changes_the_numbers(results_dir, capsys):
    argv = ["--config", str(runner.REPO_ROOT / "experimentos" / "configs" / "iris.yaml")]
    runner.main(argv)
    without = capsys.readouterr().out
    runner.main(argv + ["--show-rules"])
    with_rules = capsys.readouterr().out

    assert "Regras finais do classificador" not in without
    assert "Regras finais do classificador" in with_rules
    stable = ("Acurácia média", "OOF:", "Baseline classe", "Incerteza", "Regras candidatas", "Matriz de confusão")
    pick = lambda text: [line for line in text.splitlines() if line.startswith(stable)]  # noqa: E731
    assert pick(without) == pick(with_rules) and len(pick(without)) == len(stable)


def test_show_rules_lists_each_fold_in_priority_order(tmp_path):
    path, _ = make_dataset(tmp_path)
    run = run_experiment(small_config(path))
    text = "\n".join(runner.format_final_rules(run))
    class_column = run["summary"]["config"]["class_column"]

    for index, fold in enumerate(run["summary"]["folds"]):
        header = f"=== Dobra {index + 1} de {len(run['summary']['folds'])}: {fold['num_final_rules']} regras"
        assert header in text
    assert f"{class_column}=" not in text  # a classe aparece sem o prefixo da coluna
    block = text.split("=== Dobra 1 ")[1].split("=== Dobra 2 ")[0]
    confidences = [float(line.split("confiança ")[1].split(",")[0]) for line in block.splitlines() if "=>" in line]
    assert len(confidences) == run["summary"]["folds"][0]["num_final_rules"]
    assert confidences == sorted(confidences, reverse=True)
