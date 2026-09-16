"""Runner de experimentos: roda o pipeline completo do CBA num dataset a
partir de um arquivo de configuração YAML.

Uso:

    python -m experimentos.runner --config experimentos/configs/iris.yaml

Para cada fold de uma validação cruzada estratificada, o runner:

1. Separa treino e teste do fold.
2. Ajusta o discretizador (``EqualFrequencyDiscretizer``) **apenas** nos
   dados de treino do fold, e o aplica em treino e teste — a discretização
   nunca vê o teste antes de ser ajustada, para evitar vazamento de
   informação (ver seção 7.3 de ``docs/PIBIC___João_e_Fábio.pdf``).
3. Codifica treino e teste em transações e roda CBA-RG (``generate_cars``)
   e CBA-CB M1 (``build_classifier_m1``) usando apenas o treino.
4. Avalia o classificador resultante no teste do fold.

Ao final, grava um JSON detalhado por execução em
``experimentos/resultados/`` (ignorado pelo git) e acrescenta uma linha de
resumo a ``experimentos/resultados/resumo.csv`` (versionado), contendo:
semente, partições (tamanho de treino/teste por fold), tempo de execução,
número de regras candidatas e finais, comprimento médio do antecedente,
acurácia por fold e versão das dependências instaladas.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
import yaml

from fase2_implementacao.association_rules import dataframe_to_transactions
from fase2_implementacao.cba import build_classifier_m1, generate_cars
from fase2_implementacao.discretization import EqualFrequencyDiscretizer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experimentos" / "resultados"
DEPENDENCIES = ["pandas", "numpy", "pyyaml", "pytest"]


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    required = {
        "dataset_path",
        "class_column",
        "continuous_columns",
        "n_bins",
        "min_support",
        "min_confidence",
        "k_folds",
        "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config incompleto, faltando: {sorted(missing)}")
    return config


def stratified_kfold_indices(labels: list, k: int, seed: int) -> list[list[int]]:
    """Índices de uma validação cruzada estratificada em k folds.

    Implementação própria (sem scikit-learn): agrupa os índices por classe,
    embaralha cada grupo com uma semente fixa e distribui os exemplos de
    cada classe entre os k folds em round-robin, o que preserva
    aproximadamente a proporção de cada classe em cada fold.
    """
    if k < 2:
        raise ValueError("k_folds deve ser >= 2")
    rng = random.Random(seed)
    indices_by_class: dict = defaultdict(list)
    for i, label in enumerate(labels):
        indices_by_class[label].append(i)

    folds: list[list[int]] = [[] for _ in range(k)]
    for label in sorted(indices_by_class):
        idxs = indices_by_class[label]
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            folds[i % k].append(idx)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def dependency_versions() -> dict[str, str]:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "desconhecida"
    versions["python"] = sys.version.split()[0]
    return versions


def run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    class_column = config["class_column"]
    feature_columns = [c for c in train_df.columns if c != class_column]

    discretizer = EqualFrequencyDiscretizer(n_bins=config["n_bins"], columns=config["continuous_columns"])
    train_discretized = discretizer.fit_transform(train_df)
    test_discretized = discretizer.transform(test_df)

    columns_used = feature_columns + [class_column]
    train_transactions = dataframe_to_transactions(train_discretized[columns_used], class_column)
    test_transactions = dataframe_to_transactions(test_discretized[columns_used], class_column)
    train_labels = train_df[class_column].astype(str).tolist()
    test_labels = test_df[class_column].astype(str).tolist()

    start = time.perf_counter()
    cars = generate_cars(
        train_transactions,
        class_attribute=class_column,
        min_support=config["min_support"],
        min_confidence=config["min_confidence"],
        max_antecedent_length=config.get("max_antecedent_length"),
    )
    classifier = build_classifier_m1(cars, train_transactions, train_labels, class_attribute=class_column)
    elapsed = time.perf_counter() - start

    predictions = classifier.predict_many(test_transactions)
    accuracy = sum(p == y for p, y in zip(predictions, test_labels)) / len(test_labels)

    antecedent_lengths = [len(rule.antecedent) for rule in classifier.rules]
    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "elapsed_seconds": elapsed,
        "num_candidate_rules": len(cars),
        "num_final_rules": len(classifier.rules),
        "avg_antecedent_length": mean(antecedent_lengths) if antecedent_lengths else 0.0,
        "accuracy": accuracy,
    }


def run_experiment(config: dict) -> dict:
    dataset_path = REPO_ROOT / config["dataset_path"]
    df = pd.read_csv(dataset_path)
    if config.get("id_column") and config["id_column"] in df.columns:
        df = df.drop(columns=[config["id_column"]])

    class_column = config["class_column"]
    labels = df[class_column].astype(str).tolist()
    folds = stratified_kfold_indices(labels, config["k_folds"], config["seed"])

    fold_results = []
    total_start = time.perf_counter()
    for fold_index in range(config["k_folds"]):
        test_idx = folds[fold_index]
        train_idx = [i for i in range(len(df)) if i not in set(test_idx)]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        result = run_fold(train_df, test_df, config)
        result["fold"] = fold_index
        fold_results.append(result)
    total_elapsed = time.perf_counter() - total_start

    accuracies = [r["accuracy"] for r in fold_results]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "dependency_versions": dependency_versions(),
        "folds": fold_results,
        "accuracy_mean": mean(accuracies),
        "accuracy_std": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "num_candidate_rules_mean": mean(r["num_candidate_rules"] for r in fold_results),
        "num_final_rules_mean": mean(r["num_final_rules"] for r in fold_results),
        "avg_antecedent_length_mean": mean(r["avg_antecedent_length"] for r in fold_results),
        "total_elapsed_seconds": total_elapsed,
    }
    return summary


def write_results(summary: dict, dataset_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    detail_path = RESULTS_DIR / f"{dataset_name}_{timestamp}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_path = RESULTS_DIR / "resumo.csv"
    is_new = not summary_path.exists()
    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                [
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
                ]
            )
        writer.writerow(
            [
                summary["timestamp"],
                dataset_name,
                summary["config"]["seed"],
                summary["config"]["k_folds"],
                f"{summary['accuracy_mean']:.4f}",
                f"{summary['accuracy_std']:.4f}",
                f"{summary['num_candidate_rules_mean']:.1f}",
                f"{summary['num_final_rules_mean']:.1f}",
                f"{summary['avg_antecedent_length_mean']:.2f}",
                f"{summary['total_elapsed_seconds']:.3f}",
                detail_path.name,
            ]
        )
    return detail_path


def main(argv: list[str] | None = None) -> None:
    # Garante saída em UTF-8 mesmo em consoles Windows configurados com uma
    # codepage legada (evita caracteres acentuados corrompidos no terminal).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Roda o pipeline CBA sobre um dataset via config YAML.")
    parser.add_argument("--config", required=True, type=Path, help="Caminho do arquivo de configuração YAML.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    summary = run_experiment(config)
    dataset_name = Path(config["dataset_path"]).stem
    detail_path = write_results(summary, dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Seed: {config['seed']}  |  k_folds: {config['k_folds']}")
    print(f"Acurácia média: {summary['accuracy_mean']:.4f} (desvio padrão: {summary['accuracy_std']:.4f})")
    print(
        "Regras candidatas (média): "
        f"{summary['num_candidate_rules_mean']:.1f}  |  "
        f"Regras finais (média): {summary['num_final_rules_mean']:.1f}  |  "
        f"Comprimento médio do antecedente: {summary['avg_antecedent_length_mean']:.2f}"
    )
    print(f"Tempo total: {summary['total_elapsed_seconds']:.3f}s")
    print(f"Detalhes salvos em: {detail_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
