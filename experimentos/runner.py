"""Runner de experimentos: roda o pipeline completo do CBA num dataset a
partir de um arquivo de configuração YAML.

Uso:

    python -m experimentos.runner --config experimentos/configs/iris.yaml
    python -m experimentos.runner --config experimentos/configs/pokemon.yaml --sem-discretizar
    python -m experimentos.runner --config experimentos/configs/pokemon.yaml --split row
    python -m experimentos.runner --config experimentos/configs/pokemon.yaml --cb-variant cba_qs

Protocolo de validação
----------------------
Validação cruzada externa com ``k_folds`` dobras. Se o YAML define
``group_column`` (ex.: ``"#"`` no Pokémon, o identificador da espécie, que
agrupa formas "Mega" da mesma espécie), as dobras são construídas **por
grupo**: nenhum grupo aparece em treino e teste da mesma dobra (achado A05
do Relatório 02), com estratificação por classe feita da melhor forma
possível. Sem ``group_column`` (ou com ``--split row``) a divisão é por
linha, estratificada. O identificador/grupo nunca é usado como preditor.

Para cada dobra, o runner:

1. Separa treino e teste.
2. Se ``discretizar`` estiver ativo (config; ``--sem-discretizar`` desliga),
   ajusta o ``EqualFrequencyDiscretizer`` **apenas** no treino e o aplica a
   treino e teste. Sem discretização os valores contínuos brutos viram itens.
3. Codifica treino em transações (com a classe) e o teste **apenas com os
   preditores** — a classe verdadeira nunca participa da inferência.
4. Roda o CBA-RG (``generate_cars``) no treino e o CBA-CB da variante
   ``cb_variant`` (``"m1"``: ``build_classifier_m1``, com o corte de erro
   mínimo por prefixo; ``"cba_qs"``: variante pré-existente, congelada nesta
   baseline e fora do escopo da validação).
5. Prediz o teste registrando, por caso, a classe prevista e a regra acionada.

Artefatos por execução
----------------------
Cada execução recebe um ``run_id`` único (dataset, discretização, divisão,
data/hora com microssegundos e sufixo aleatório) e grava, sem sobrescrever
nada, o diretório ``experimentos/resultados/<run_id>/`` (ignorado pelo git)
com: ``summary.json``, ``config.json`` (config normalizada, seed, commit,
hash do dataset), ``environment.txt`` (equivalente a ``pip freeze``),
``partitions.json``, ``discretization.json``, ``rules.json``,
``m1_history.json``, ``oof_predictions.csv`` e ``metrics.json``. Além disso,
acrescenta uma linha a ``experimentos/resultados/resumo.csv`` (versionado),
migrando o cabeçalho de arquivos antigos quando necessário.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
import yaml

from experimentos.metricas import METRIC_KEYS, bootstrap_ci, compute_metrics
from fase2_implementacao.association_rules import dataframe_to_feature_transactions, dataframe_to_transactions
from fase2_implementacao.cba import build_classifier_m1, generate_cars, majority_class, sort_cars_by_precedence
from fase2_implementacao.cba_qs import DEFAULT_WEIGHTS as CBA_QS_DEFAULT_WEIGHTS
from fase2_implementacao.cba_qs import build_classifier_cba_qs
from fase2_implementacao.discretization import EqualFrequencyDiscretizer
from fase2_implementacao.discretization_mdl import MDLDiscretizer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experimentos" / "resultados"

SUMMARY_COLUMNS = [
    "timestamp",
    "dataset",
    "seed",
    "k_folds",
    "accuracy_mean",
    "accuracy_std",
    "num_candidate_rules_mean",
    "num_final_rules_mean",
    "avg_antecedent_length_mean",
    "total_elapsed_seconds",
    "detail_file",
    "discretizar",
    "default_class_usage_rate_mean",
    "recall_positive_class_mean",
    "cb_variant",
    # colunas acrescentadas ao final (compatibilidade posicional com linhas antigas)
    "run_id",
    "split",
    "oof_accuracy",
    "oof_balanced_accuracy",
    "oof_macro_f1",
    "oof_mcc",
    "oof_recall_positive_class",
    "majority_baseline_accuracy",
    "discretization_method",
    "prune",
]


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = {
        "dataset_path",
        "class_column",
        "continuous_columns",
        "min_support",
        "min_confidence",
        "k_folds",
        "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config incompleto, faltando: {sorted(missing)}")
    config.setdefault("discretizar", True)
    config.setdefault("discretization_method", "quantile")  # "quantile" (baseline) ou "mdl" (Fayyad e Irani, como no artigo)
    config.setdefault("n_bins", 3)  # só usado por "quantile"
    config.setdefault("prune", False)  # poda pessimista de condições (opcional no artigo)
    config.setdefault("prune_cf", 0.25)
    config.setdefault("csv_sep", ",")
    config.setdefault("feature_columns", None)
    config.setdefault("positive_class", None)
    config.setdefault("id_column", None)
    config.setdefault("group_column", None)
    config.setdefault("max_antecedent_length", None)
    config.setdefault("cb_variant", "m1")
    config.setdefault("cba_qs_weights", None)
    return validate_config(config)


def validate_config(config: dict) -> dict:
    """Valida tipos/faixas antes de qualquer treino e normaliza ``positive_class`` (A09).

    ``positive_class`` passa pelo mesmo contrato do alvo (``str``): o ``true``
    do YAML vira ``"True"``, igual ao rótulo ``"True"`` de uma coluna booleana.
    As chaves opcionais da discretização supervisionada e da poda recebem o padrão
    (baseline: quantis, sem poda) quando ausentes, para que configurações antigas sigam válidas.
    """
    config.setdefault("discretization_method", "quantile")
    config.setdefault("n_bins", 3)
    config.setdefault("prune", False)
    config.setdefault("prune_cf", 0.25)
    if config["cb_variant"] not in ("m1", "cba_qs"):
        raise ValueError(f"cb_variant deve ser 'm1' ou 'cba_qs', recebido: {config['cb_variant']!r}")
    if not _is_real(config["min_support"]) or not 0 < config["min_support"] <= 1:
        raise ValueError(f"min_support deve ser numérico finito em (0, 1], recebido: {config['min_support']!r}")
    if not _is_real(config["min_confidence"]) or not 0 <= config["min_confidence"] <= 1:
        raise ValueError(f"min_confidence deve ser numérico finito em [0, 1], recebido: {config['min_confidence']!r}")
    if not _is_int(config["n_bins"]) or config["n_bins"] < 2:
        raise ValueError(f"n_bins deve ser inteiro >= 2, recebido: {config['n_bins']!r}")
    mal = config["max_antecedent_length"]
    if mal is not None and (not _is_int(mal) or mal < 1):
        raise ValueError(f"max_antecedent_length deve ser None ou inteiro >= 1, recebido: {mal!r}")
    if not _is_int(config["k_folds"]) or config["k_folds"] < 2:
        raise ValueError(f"k_folds deve ser inteiro >= 2, recebido: {config['k_folds']!r}")
    if not _is_int(config["seed"]):
        raise ValueError(f"seed deve ser inteiro, recebido: {config['seed']!r}")
    if not isinstance(config["discretizar"], bool):
        raise ValueError(f"discretizar deve ser booleano, recebido: {config['discretizar']!r}")
    if config["discretization_method"] not in ("quantile", "mdl"):
        raise ValueError(f"discretization_method deve ser 'quantile' ou 'mdl', recebido: {config['discretization_method']!r}")
    if not isinstance(config["prune"], bool):
        raise ValueError(f"prune deve ser booleano, recebido: {config['prune']!r}")
    if not _is_real(config["prune_cf"]) or not 0 < config["prune_cf"] < 1:
        raise ValueError(f"prune_cf deve estar em (0, 1), recebido: {config['prune_cf']!r}")
    if not isinstance(config["class_column"], str) or not config["class_column"]:
        raise ValueError("class_column deve ser o nome de uma coluna")

    features = config["feature_columns"]
    if features is not None:
        if len(set(features)) != len(features):
            raise ValueError("feature_columns contém colunas duplicadas")
        if config["class_column"] in features:
            raise ValueError("a coluna-alvo não pode estar em feature_columns")
        for key in ("id_column", "group_column"):
            if config[key] is not None and config[key] in features:
                raise ValueError(f"{key} ({config[key]!r}) não pode ser usado como preditor")
    continuous = config["continuous_columns"] or []
    if config["class_column"] in continuous:
        raise ValueError("a coluna-alvo não pode ser discretizada (continuous_columns)")
    if features is not None and not set(continuous) <= set(features):
        raise ValueError(f"continuous_columns deve ser subconjunto de feature_columns: {sorted(set(continuous) - set(features))}")

    if config["positive_class"] is not None:
        config["positive_class"] = str(config["positive_class"])
    return config


# --------------------------------------------------------------------------- #
# Particionamento
# --------------------------------------------------------------------------- #


def _validate_k(k, n: int) -> None:
    if not _is_int(k) or k < 2:
        raise ValueError(f"k_folds deve ser inteiro >= 2, recebido: {k!r}")
    if k > n:
        raise ValueError(f"k_folds ({k}) não pode exceder o número de exemplos ({n})")


def stratified_kfold_indices(labels: list, k: int, seed: int) -> list[list[int]]:
    """Índices de uma validação cruzada estratificada em k folds.

    Implementação própria (sem scikit-learn): agrupa os índices por classe,
    embaralha cada grupo com uma semente fixa e distribui os exemplos em
    round-robin com um **ponteiro contínuo entre classes** (o resto de uma
    classe continua de onde a anterior parou). Assim cada classe fica
    equilibrada (±1 exemplo) entre as dobras e o tamanho total das dobras
    também (±1), sem acumular os restos nas primeiras dobras (achado A06).

    Rejeita ``k`` inválido, ``k`` maior que o nº de exemplos e ``k`` maior
    que o tamanho da menor classe (a estratificação seria inviável).
    """
    _validate_k(k, len(labels))
    indices_by_class: dict = defaultdict(list)
    for i, label in enumerate(labels):
        indices_by_class[label].append(i)
    smallest = min(len(v) for v in indices_by_class.values())
    if k > smallest:
        raise ValueError(
            f"k_folds ({k}) excede o tamanho da menor classe ({smallest}): a estratificação seria inviável; reduza k_folds"
        )

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    pointer = 0
    for label in sorted(indices_by_class):
        idxs = indices_by_class[label]
        rng.shuffle(idxs)
        for idx in idxs:
            folds[pointer % k].append(idx)
            pointer += 1
    for fold in folds:
        rng.shuffle(fold)
    return folds


def stratified_group_kfold_indices(labels: list, groups: list, k: int, seed: int) -> list[list[int]]:
    """Validação cruzada **por grupo** (nenhum grupo em duas dobras), estratificada por classe.

    Guloso: grupos grandes primeiro (empates embaralhados com ``seed``); cada
    grupo vai para a dobra cujo desvio quadrático em relação às contagens
    alvo por classe (``total_da_classe / k``) menos aumenta, desempatando por
    menor tamanho e depois menor índice. Rejeita ``k`` maior que o nº de
    grupos e dobras vazias. A estratificação é aproximada (um grupo pode ter
    mais de uma linha, e pode misturar classes).
    """
    if len(labels) != len(groups):
        raise ValueError("labels e groups devem ter o mesmo tamanho")
    _validate_k(k, len(labels))
    members: dict = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    if k > len(members):
        raise ValueError(f"k_folds ({k}) excede o número de grupos ({len(members)})")

    classes = sorted(set(labels))
    totals = Counter(labels)
    target = {c: totals[c] / k for c in classes}
    group_counts = {g: Counter(labels[i] for i in idxs) for g, idxs in members.items()}

    rng = random.Random(seed)
    order = list(members)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))  # estável: empates mantêm a ordem embaralhada

    fold_counts = [Counter() for _ in range(k)]
    fold_sizes = [0] * k
    fold_groups: list[list] = [[] for _ in range(k)]
    for g in order:
        best_fold, best_key = 0, None
        for f in range(k):
            delta = sum(
                ((fold_counts[f][c] + group_counts[g][c] - target[c]) ** 2 - (fold_counts[f][c] - target[c]) ** 2) / target[c]
                for c in classes
            )
            key = (delta, fold_sizes[f], f)
            if best_key is None or key < best_key:
                best_fold, best_key = f, key
        fold_counts[best_fold].update(group_counts[g])
        fold_sizes[best_fold] += len(members[g])
        fold_groups[best_fold].append(g)

    folds = [[i for g in fg for i in members[g]] for fg in fold_groups]
    if any(not fold for fold in folds):
        raise ValueError("a divisão por grupos gerou uma dobra vazia; reduza k_folds")
    for fold in folds:
        rng.shuffle(fold)
    return folds


# --------------------------------------------------------------------------- #
# Proveniência / ambiente
# --------------------------------------------------------------------------- #


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in ["pandas", "numpy", "pyyaml", "pytest"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "desconhecida"
    versions["python"] = sys.version.split()[0]
    return versions


def environment_snapshot() -> str:
    """Lista ``nome==versão`` de todos os pacotes instalados (equivalente ao ``pip freeze``)."""
    lines = sorted(
        {f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata["Name"]},
        key=str.lower,
    )
    header = [f"# python {sys.version.split()[0]} | {platform.platform()}"]
    return "\n".join(header + lines) + "\n"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #


def _rule_to_dict(rule) -> dict:
    return {
        "antecedent": sorted(f"{a}={v}" for a, v in rule.antecedent),
        "consequent": f"{rule.consequent[0]}={rule.consequent[1]}",
        "joint_count": rule.joint_count,
        "antecedent_count": rule.antecedent_count,
        "n_transactions": rule.n_transactions,
        "confidence": f"{rule.joint_count}/{rule.antecedent_count}",
        "confidence_float": rule.confidence,
        "support_float": rule.support,
        "generation_order": rule.generation_order,
    }


def _rule_text(rule) -> str:
    return " AND ".join(sorted(f"{a}={v}" for a, v in rule.antecedent)) + f" => {rule.consequent[0]}={rule.consequent[1]}"


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> dict:
    """Roda um fold. ``train_df``/``test_df`` contêm só preditores + coluna de classe."""
    class_column = config["class_column"]
    feature_columns = [c for c in train_df.columns if c != class_column]
    t0 = time.perf_counter()

    discretization = None
    if config["discretizar"]:
        if config["discretization_method"] == "mdl":
            # supervisionada: cortes aprendidos com X e y do TREINO; a transformação não usa o alvo
            discretizer = MDLDiscretizer(columns=config["continuous_columns"])
            train_encoded = discretizer.fit_transform(train_df, train_df[class_column].astype(str))
        else:
            discretizer = EqualFrequencyDiscretizer(n_bins=config["n_bins"], columns=config["continuous_columns"])
            train_encoded = discretizer.fit_transform(train_df)
        test_encoded = discretizer.transform(test_df[feature_columns])
        discretization = discretizer.describe(train_df)
    else:
        # Sem discretização: cada valor bruto distinto vira seu próprio item.
        train_encoded = train_df
        test_encoded = test_df[feature_columns]

    train_transactions = dataframe_to_transactions(train_encoded, class_column)
    test_transactions = dataframe_to_feature_transactions(test_encoded)  # sem a classe verdadeira
    train_labels = train_df[class_column].astype(str).tolist()
    test_labels = test_df[class_column].astype(str).tolist()
    t_prep = time.perf_counter()

    cars = generate_cars(
        train_transactions,
        class_attribute=class_column,
        min_support=config["min_support"],
        min_confidence=config["min_confidence"],
        max_antecedent_length=config["max_antecedent_length"],
        prune=config["prune"],
        prune_cf=config["prune_cf"],
    )
    t_rg = time.perf_counter()

    if config["cb_variant"] == "cba_qs":
        weights = tuple(config["cba_qs_weights"]) if config["cba_qs_weights"] else CBA_QS_DEFAULT_WEIGHTS
        classifier = build_classifier_cba_qs(
            cars, train_transactions, train_labels, class_attribute=class_column, weights=weights
        )
    else:
        classifier = build_classifier_m1(cars, train_transactions, train_labels, class_attribute=class_column)
    t_cb = time.perf_counter()

    predictions = []
    for transaction in test_transactions:
        label, rule_index = classifier.predict_with_rule(transaction)
        predictions.append((label, rule_index))
    t_inf = time.perf_counter()

    y_pred = [p for p, _ in predictions]
    metrics = compute_metrics(test_labels, y_pred, labels=sorted(set(train_labels) | set(test_labels)))
    default_usage = sum(1 for _, idx in predictions if idx is None)

    result = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "elapsed_seconds": t_cb - t_prep,  # CBA-RG + CBA-CB (compatível com versões anteriores)
        "timing_seconds": {
            "preprocessing": t_prep - t0,
            "rule_generation": t_rg - t_prep,
            "classifier_building": t_cb - t_rg,
            "inference": t_inf - t_cb,
        },
        "num_candidate_rules": len(cars),
        "num_final_rules": len(classifier.rules),
        "avg_antecedent_length": mean(len(rule.antecedent) for rule in classifier.rules) if classifier.rules else 0.0,
        "accuracy": metrics["accuracy"],
        "default_class_usage": default_usage,
        "default_class_usage_rate": default_usage / len(test_labels),
        "metrics": metrics,
        "majority_baseline_class": majority_class(train_labels),
        "default_class": classifier.default_class,
        "m1_chosen_prefix_length": classifier.chosen_prefix_length,
        "_predictions": predictions,
        "_test_labels": test_labels,
        "_discretization": discretization,
        "_classifier": classifier,
        "_cars": cars,
    }
    positive_class = config["positive_class"]
    if positive_class is not None:
        result["recall_positive_class"] = metrics["per_class"].get(positive_class, {}).get("recall")
    return result


def _rules_artifact(result: dict, config: dict) -> list[dict]:
    """Regras candidatas do fold, com precedência, estado no M1 e posição final."""
    classifier = result["_classifier"]
    cars = result["_cars"]
    final_position = {rule: i for i, rule in enumerate(classifier.rules)}
    selected_before_cut = {record.rule: record.n_rules - 1 for record in classifier.m1_history}
    precedence = (
        {rule: i for i, rule in enumerate(sort_cars_by_precedence(cars))} if config["cb_variant"] == "m1" else {}
    )
    rows = []
    for rule in cars:
        if rule in final_position:
            status = "final"
        elif rule in selected_before_cut:
            status = "selected_then_cut_by_m1"
        else:
            status = "not_selected_by_coverage"
        row = _rule_to_dict(rule)
        row.update(
            {
                "precedence_position": precedence.get(rule),
                "status": status,
                "selection_position": selected_before_cut.get(rule),
                "final_position": final_position.get(rule),
                "text": _rule_text(rule),
            }
        )
        rows.append(row)
    return rows


def _history_artifact(result: dict) -> dict:
    classifier = result["_classifier"]
    return {
        "chosen_prefix_length": classifier.chosen_prefix_length,
        "final_default_class": classifier.default_class,
        "prefixes": [
            {
                "n_rules": r.n_rules,
                "rule": _rule_text(r.rule),
                "rule_errors": r.rule_errors,
                "n_remaining": r.n_remaining,
                "default_class": r.default_class,
                "default_errors": r.default_errors,
                "total_errors": r.total_errors,
                "chosen": r.n_rules == classifier.chosen_prefix_length,
            }
            for r in classifier.m1_history
        ],
    }


def run_experiment(config: dict) -> dict:
    config = validate_config(copy.deepcopy(config))
    dataset_path = REPO_ROOT / config["dataset_path"]
    df = pd.read_csv(dataset_path, sep=config["csv_sep"])

    class_column = config["class_column"]
    id_column, group_column = config["id_column"], config["group_column"]
    for name in (class_column, id_column, group_column):
        if name is not None and name not in df.columns:
            raise ValueError(f"coluna {name!r} não encontrada em {config['dataset_path']}")
    if df[class_column].isna().any():
        raise ValueError(f"a coluna-alvo {class_column!r} contém valores ausentes")

    if config["feature_columns"] is None:
        config["feature_columns"] = [c for c in df.columns if c not in (class_column, id_column, group_column)]
    feature_columns = config["feature_columns"]
    missing = [c for c in feature_columns + list(config["continuous_columns"] or []) if c not in df.columns]
    if missing:
        raise ValueError(f"colunas inexistentes no dataset: {sorted(set(missing))}")
    if not set(config["continuous_columns"] or []) <= set(feature_columns):
        raise ValueError("continuous_columns deve ser subconjunto de feature_columns")
    if group_column is not None and group_column in feature_columns:
        raise ValueError("group_column não pode ser usado como preditor")

    labels = df[class_column].astype(str).tolist()
    if config["positive_class"] is not None and config["positive_class"] not in set(labels):
        raise ValueError(
            f"positive_class {config['positive_class']!r} não existe entre as classes do dataset: {sorted(set(labels))}"
        )

    row_ids = list(range(len(df)))
    ids = df[id_column].tolist() if id_column is not None else row_ids
    groups = df[group_column].astype(str).tolist() if group_column is not None else None
    split = config.get("split") or ("group" if groups is not None else "row")
    if split == "group" and groups is None:
        raise ValueError("split='group' exige group_column no config")
    config["split"] = split

    if split == "group":
        folds = stratified_group_kfold_indices(labels, groups, config["k_folds"], config["seed"])
    else:
        folds = stratified_kfold_indices(labels, config["k_folds"], config["seed"])

    model_df = df[feature_columns + [class_column]]
    fold_results, oof_rows, partitions, discretizations, rules_art, history_art = [], [], [], {}, {}, {}
    seen_test = Counter()
    total_start = time.perf_counter()
    for fold_index, test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = [i for i in row_ids if i not in test_set]
        train_df = model_df.iloc[train_idx].reset_index(drop=True)
        test_df = model_df.iloc[test_idx].reset_index(drop=True)
        result = run_fold(train_df, test_df, config)
        result["fold"] = fold_index

        classifier = result["_classifier"]
        for row_pos, (label, rule_index) in zip(test_idx, result["_predictions"]):
            seen_test[row_pos] += 1
            oof_rows.append(
                {
                    "fold": fold_index,
                    "row_id": row_pos,
                    "id": ids[row_pos],
                    "group": groups[row_pos] if groups is not None else "",
                    "y_true": labels[row_pos],
                    "y_pred": label,
                    "rule_index": "" if rule_index is None else rule_index,
                    "rule": "" if rule_index is None else _rule_text(classifier.rules[rule_index]),
                    "used_default": int(rule_index is None),
                    "correct": int(label == labels[row_pos]),
                }
            )

        part = {
            "fold": fold_index,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "test_row_ids": sorted(test_idx),
            "test_ids": [ids[i] for i in sorted(test_idx)],
            "train_class_counts": dict(Counter(labels[i] for i in train_idx)),
            "test_class_counts": dict(Counter(labels[i] for i in test_idx)),
        }
        part["classes_missing_in_test"] = sorted(set(labels) - set(part["test_class_counts"]))
        if groups is not None:
            train_groups, test_groups = {groups[i] for i in train_idx}, {groups[i] for i in test_idx}
            part["train_groups"], part["test_groups"] = len(train_groups), len(test_groups)
            part["groups_shared_train_test"] = len(train_groups & test_groups)
            result["groups_shared_train_test"] = len(train_groups & test_groups)
        partitions.append(part)
        discretizations[fold_index] = result["_discretization"]
        rules_art[fold_index] = _rules_artifact(result, config)
        history_art[fold_index] = _history_artifact(result)
        fold_results.append(result)
    total_elapsed = time.perf_counter() - total_start

    if any(seen_test[i] != 1 for i in row_ids):
        raise RuntimeError("falha de partição: alguma linha não aparece exatamente uma vez nos testes")

    y_true_all = [r["y_true"] for r in oof_rows]
    y_pred_all = [r["y_pred"] for r in oof_rows]
    class_labels = sorted(set(labels))
    oof_metrics = compute_metrics(y_true_all, y_pred_all, labels=class_labels)
    majority_pred = [fold_results[r["fold"]]["majority_baseline_class"] for r in oof_rows]
    majority_metrics = compute_metrics(y_true_all, majority_pred, labels=class_labels)

    # Dois tipos de variabilidade, reportados separadamente (achado A11):
    # dispersão entre folds (descritiva) e IC bootstrap das métricas OOF
    # (incerteza de generalização), reamostrando grupos quando existem.
    unit_ids = groups if groups is not None else row_ids
    uncertainty = {
        "between_folds": {
            key: {
                "mean": mean(r["metrics"][key] for r in fold_results),
                "std": pstdev([r["metrics"][key] for r in fold_results]),
                "min": min(r["metrics"][key] for r in fold_results),
                "max": max(r["metrics"][key] for r in fold_results),
            }
            for key in METRIC_KEYS
        },
        "bootstrap": {
            "unit": "group" if groups is not None else "row",
            "reps": 1000,
            "level": 0.95,
            "seed": config["seed"],
            "ci": bootstrap_ci(
                [labels[r["row_id"]] for r in oof_rows],
                [r["y_pred"] for r in oof_rows],
                [unit_ids[r["row_id"]] for r in oof_rows],
                class_labels,
                reps=1000,
                seed=config["seed"],
            ),
        },
    }

    accuracies = [r["accuracy"] for r in fold_results]
    recalls = [r["recall_positive_class"] for r in fold_results if r.get("recall_positive_class") is not None]
    positive = config["positive_class"]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "discretizar": config["discretizar"],
        "discretization_method": config["discretization_method"],
        "prune": config["prune"],
        "cb_variant": config["cb_variant"],
        "split": split,
        "config": config,
        "dependency_versions": dependency_versions(),
        "folds": [{k: v for k, v in r.items() if not k.startswith("_")} for r in fold_results],
        "accuracy_mean": mean(accuracies),
        "accuracy_std": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "num_candidate_rules_mean": mean(r["num_candidate_rules"] for r in fold_results),
        "num_final_rules_mean": mean(r["num_final_rules"] for r in fold_results),
        "avg_antecedent_length_mean": mean(r["avg_antecedent_length"] for r in fold_results),
        "default_class_usage_total": sum(r["default_class_usage"] for r in fold_results),
        "default_class_usage_rate_mean": mean(r["default_class_usage_rate"] for r in fold_results),
        "recall_positive_class_mean": mean(recalls) if recalls else None,
        "oof_metrics": oof_metrics,
        "oof_recall_positive_class": oof_metrics["per_class"][positive]["recall"] if positive else None,
        "majority_baseline_metrics": majority_metrics,
        "oof_uncertainty": uncertainty,
        "total_elapsed_seconds": total_elapsed,
    }
    artifacts = {
        "partitions": partitions,
        "discretization": discretizations,
        "rules": rules_art,
        "m1_history": history_art,
        "oof": oof_rows,
        "metrics": {
            "oof": oof_metrics,
            "majority_baseline": majority_metrics,
            "per_fold": [{"fold": r["fold"], **r["metrics"]} for r in fold_results],
        },
    }
    return {"summary": summary, "artifacts": artifacts, "dataset_path": dataset_path}


def _new_run_dir(dataset_name: str, summary: dict) -> tuple[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    disc = "sem_discretizacao" if not summary["discretizar"] else ("mdl" if summary["discretization_method"] == "mdl" else "discretizado")
    if summary["prune"]:
        disc += "_poda"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{dataset_name}_{disc}_{summary['split']}_{summary['cb_variant']}_{stamp}_{uuid.uuid4().hex[:6]}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(exist_ok=False)  # nunca sobrescreve uma execução anterior
    return run_id, run_dir


def _dump(path: Path, obj) -> None:
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def migrate_summary_header(summary_path: Path) -> None:
    """Se ``resumo.csv`` tem cabeçalho antigo (prefixo do atual), reescreve com o novo, preenchendo vazios."""
    if not summary_path.exists():
        return
    with open(summary_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0] == SUMMARY_COLUMNS:
        return
    old = rows[0]
    if old != SUMMARY_COLUMNS[: len(old)]:
        raise ValueError(f"cabeçalho de {summary_path.name} incompatível com o esquema atual: {old}")
    migrated = [SUMMARY_COLUMNS] + [row + [""] * (len(SUMMARY_COLUMNS) - len(row)) for row in rows[1:]]
    tmp = summary_path.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(migrated)
    tmp.replace(summary_path)


def write_results(run: dict, dataset_name: str) -> Path:
    """Grava todos os artefatos da execução e a linha de ``resumo.csv``; devolve ``summary.json``."""
    summary, artifacts = run["summary"], run["artifacts"]
    run_id, run_dir = _new_run_dir(dataset_name, summary)
    summary["run_id"] = run_id

    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    provenance = {
        "run_id": run_id,
        "config": summary["config"],
        "seed": summary["config"]["seed"],
        "split": summary["split"],
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "dataset_sha256": file_sha256(run["dataset_path"]),
        "dependency_versions": summary["dependency_versions"],
        "argv": sys.argv,
    }
    _dump(run_dir / "config.json", provenance)
    with open(run_dir / "environment.txt", "x", encoding="utf-8") as f:
        f.write(environment_snapshot())
    _dump(run_dir / "partitions.json", artifacts["partitions"])
    _dump(run_dir / "discretization.json", artifacts["discretization"])
    _dump(run_dir / "rules.json", artifacts["rules"])
    _dump(run_dir / "m1_history.json", artifacts["m1_history"])
    _dump(run_dir / "metrics.json", artifacts["metrics"])
    oof_columns = ["fold", "row_id", "id", "group", "y_true", "y_pred", "rule_index", "rule", "used_default", "correct"]
    with open(run_dir / "oof_predictions.csv", "x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=oof_columns)
        writer.writeheader()
        writer.writerows(artifacts["oof"])
    _dump(run_dir / "summary.json", summary)
    detail_path = run_dir / "summary.json"

    summary_path = RESULTS_DIR / "resumo.csv"
    migrate_summary_header(summary_path)
    is_new = not summary_path.exists()
    positive = summary["config"]["positive_class"]
    oof = summary["oof_metrics"]

    def fmt(value, spec=".4f"):
        return "" if value is None else format(value, spec)

    row = {
        "timestamp": summary["timestamp"],
        "dataset": dataset_name,
        "seed": summary["config"]["seed"],
        "k_folds": summary["config"]["k_folds"],
        "accuracy_mean": fmt(summary["accuracy_mean"]),
        "accuracy_std": fmt(summary["accuracy_std"]),
        "num_candidate_rules_mean": fmt(summary["num_candidate_rules_mean"], ".1f"),
        "num_final_rules_mean": fmt(summary["num_final_rules_mean"], ".1f"),
        "avg_antecedent_length_mean": fmt(summary["avg_antecedent_length_mean"], ".2f"),
        "total_elapsed_seconds": fmt(summary["total_elapsed_seconds"], ".3f"),
        "detail_file": f"{run_id}/summary.json",
        "discretizar": summary["discretizar"],
        "default_class_usage_rate_mean": fmt(summary["default_class_usage_rate_mean"]),
        "recall_positive_class_mean": fmt(summary["recall_positive_class_mean"]),
        "cb_variant": summary["cb_variant"],
        "run_id": run_id,
        "split": summary["split"],
        "oof_accuracy": fmt(oof["accuracy"]),
        "oof_balanced_accuracy": fmt(oof["balanced_accuracy"]),
        "oof_macro_f1": fmt(oof["macro_f1"]),
        "oof_mcc": fmt(oof["mcc"]),
        "oof_recall_positive_class": fmt(summary["oof_recall_positive_class"]) if positive else "",
        "majority_baseline_accuracy": fmt(summary["majority_baseline_metrics"]["accuracy"]),
        "discretization_method": summary["discretization_method"],
        "prune": summary["prune"],
    }
    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return detail_path


def format_final_rules(run: dict) -> list[str]:
    """Linhas de texto com as regras finais de cada dobra, na ordem de prioridade do classificador.

    Só formata o que ``run_experiment`` já calculou (``artifacts["rules"]``); não recalcula nada.
    """
    class_prefix = f"{run['summary']['config']['class_column']}="
    n_folds = len(run["summary"]["folds"])
    lines = ["", "Regras finais do classificador (a primeira regra que casa decide; se nenhuma casar, vale a classe padrão):"]
    for fold_index, fold in enumerate(run["summary"]["folds"]):
        final = sorted(
            (r for r in run["artifacts"]["rules"][fold_index] if r["final_position"] is not None),
            key=lambda r: r["final_position"],
        )
        lines.append("")
        lines.append(f"=== Dobra {fold_index + 1} de {n_folds}: {len(final)} regras | classe padrão: {fold['default_class']} ===")
        for r in final:
            antecedent = " AND ".join(r["antecedent"])
            consequent = r["consequent"].removeprefix(class_prefix)
            lines.append(
                f"{r['final_position'] + 1:>3}. {antecedent} => {consequent} "
                f"(confiança {r['confidence_float']:.4f}, suporte {r['support_float']:.4f})"
            )
    return lines


def main(argv: list[str] | None = None) -> None:
    # Garante saída em UTF-8 mesmo em consoles Windows configurados com uma
    # codepage legada (evita caracteres acentuados corrompidos no terminal).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Roda o pipeline CBA sobre um dataset via config YAML.")
    parser.add_argument("--config", required=True, type=Path, help="Caminho do arquivo de configuração YAML.")
    parser.add_argument(
        "--sem-discretizar",
        action="store_true",
        help="Pula a etapa de discretização, sobrescrevendo 'discretizar' do config para false.",
    )
    parser.add_argument(
        "--discretization",
        choices=["quantile", "mdl"],
        help="Sobrescreve 'discretization_method': 'quantile' (baseline) ou 'mdl' (entropia/MDL de Fayyad e Irani, como no artigo).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Liga a poda pessimista de condições (opcional no artigo; ver cba.prune_cars_pessimistic).",
    )
    parser.add_argument(
        "--cb-variant",
        choices=["m1", "cba_qs"],
        help="Sobrescreve 'cb_variant' do config: 'm1' (CBA-CB clássico) ou 'cba_qs' (congelado nesta baseline).",
    )
    parser.add_argument(
        "--split",
        choices=["group", "row"],
        help="Protocolo de divisão: 'group' (por group_column) ou 'row' (por linha). Padrão: group se o config define group_column.",
    )
    parser.add_argument(
        "--show-rules",
        action="store_true",
        help="Depois do resultado, imprime as regras finais de cada dobra (só visibilidade: não altera nenhum número).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.sem_discretizar:
        config["discretizar"] = False
    if args.discretization:
        config["discretization_method"] = args.discretization
    if args.prune:
        config["prune"] = True
    if args.cb_variant:
        config["cb_variant"] = args.cb_variant
    if args.split:
        config["split"] = args.split

    run = run_experiment(config)
    summary = run["summary"]
    dataset_name = Path(config["dataset_path"]).stem
    detail_path = write_results(run, dataset_name)

    oof, base = summary["oof_metrics"], summary["majority_baseline_metrics"]
    print(f"Dataset: {dataset_name}  |  run_id: {summary['run_id']}")
    print(
        f"Discretização: {summary['discretization_method'] if summary['discretizar'] else 'não'}  |  "
        f"Poda pessimista: {'sim' if summary['prune'] else 'não'}  |  Divisão: {summary['split']}"
    )
    print(f"Variante CBA-CB: {summary['cb_variant']}")
    print(f"Seed: {config['seed']}  |  k_folds: {config['k_folds']}")
    print(f"Acurácia média por dobra: {summary['accuracy_mean']:.4f} (desvio padrão: {summary['accuracy_std']:.4f})")
    print(
        f"OOF: acurácia {oof['accuracy']:.4f} | acurácia balanceada {oof['balanced_accuracy']:.4f} | "
        f"macro-F1 {oof['macro_f1']:.4f} | MCC {oof['mcc']:.4f}"
    )
    print(f"Baseline classe majoritária (OOF): acurácia {base['accuracy']:.4f} | MCC {base['mcc']:.4f}")
    unc = summary["oof_uncertainty"]
    ci = unc["bootstrap"]["ci"]
    print(
        f"Incerteza (IC 95% bootstrap por {unc['bootstrap']['unit']}): acurácia [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}] | "
        f"MCC [{ci['mcc'][0]:.4f}, {ci['mcc'][1]:.4f}]  ||  dispersão entre folds (desvio): "
        f"acurácia {unc['between_folds']['accuracy']['std']:.4f} | MCC {unc['between_folds']['mcc']['std']:.4f}"
    )
    print(
        "Regras candidatas (média): "
        f"{summary['num_candidate_rules_mean']:.1f}  |  "
        f"Regras finais (média): {summary['num_final_rules_mean']:.1f}  |  "
        f"Comprimento médio do antecedente: {summary['avg_antecedent_length_mean']:.2f}"
    )
    print(
        "Uso da classe padrão no teste (média): "
        f"{summary['default_class_usage_rate_mean']:.4f}  |  "
        f"total de casos: {summary['default_class_usage_total']}"
    )
    if summary["recall_positive_class_mean"] is not None:
        print(
            f"Recall da classe positiva ({config['positive_class']!r}): média por dobra "
            f"{summary['recall_positive_class_mean']:.4f} | OOF {summary['oof_recall_positive_class']:.4f}"
        )
    print(f"Matriz de confusão OOF {oof['confusion_matrix']['labels']}: {oof['confusion_matrix']['matrix']}")
    print(f"Tempo total: {summary['total_elapsed_seconds']:.3f}s")
    shown = detail_path.parent
    try:
        shown = shown.relative_to(REPO_ROOT)
    except ValueError:  # diretório de resultados fora do repositório
        pass
    print(f"Detalhes salvos em: {shown}")
    if args.show_rules:
        print("\n".join(format_final_rules(run)))


if __name__ == "__main__":
    main()
