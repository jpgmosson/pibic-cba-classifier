"""Runner de experimentos: roda o pipeline completo do CBA num dataset a
partir de um arquivo de configuração YAML.

Uso:

    python -m experimentos.runner --config experimentos/configs/iris.yaml
    python -m experimentos.runner --config experimentos/configs/pokemon.yaml --sem-discretizar

Para cada fold de uma validação cruzada estratificada, o runner:

1. Separa treino e teste do fold.
2. Se ``discretizar`` (config, padrão true) estiver ativo — e a flag de CLI
   ``--sem-discretizar`` não tiver sido passada —, ajusta o discretizador
   (``EqualFrequencyDiscretizer``) **apenas** nos dados de treino do fold, e
   o aplica em treino e teste — a discretização nunca vê o teste antes de
   ser ajustada, para evitar vazamento de informação (ver seção 7.3 de
   ``docs/PIBIC___João_e_Fábio.pdf``). Se ``discretizar`` for false, os
   atributos contínuos entram crus na codificação transacional (cada valor
   numérico distinto vira seu próprio item) — a lógica de CBA-RG/CBA-CB não
   muda, só a etapa de preparação dos dados é pulada.
3. Codifica treino e teste em transações e roda CBA-RG (``generate_cars``)
   e CBA-CB M1 (``build_classifier_m1``) usando apenas o treino.
4. Avalia o classificador resultante no teste do fold: acurácia e, se
   ``positive_class`` estiver definido no config, recall dessa classe e
   quantas vezes a classe padrão foi usada (cobertura do classificador).

Ao final, grava um JSON detalhado por execução em
``experimentos/resultados/`` (ignorado pelo git; o nome do arquivo indica se
a execução usou discretização) e acrescenta uma linha de resumo a
``experimentos/resultados/resumo.csv`` (versionado), contendo: se
discretizou, semente, partições (tamanho de treino/teste por fold), tempo de
execução, número de regras candidatas e finais, comprimento médio do
antecedente, acurácia por fold, uso da classe padrão, recall da classe
positiva (quando configurada) e versão das dependências instaladas.
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
    config.setdefault("discretizar", True)
    config.setdefault("csv_sep", ",")
    config.setdefault("feature_columns", None)
    config.setdefault("positive_class", None)
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


def _count_default_usage(classifier, transactions) -> int:
    """Conta quantas transações não são cobertas por nenhuma regra do classificador
    (isto é, quantas vezes a predição recai na classe padrão)."""
    return sum(1 for t in transactions if not any(rule.matches(t) for rule in classifier.rules))


def run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> dict:
    class_column = config["class_column"]
    feature_columns = config["feature_columns"] or [c for c in train_df.columns if c != class_column]
    columns_used = feature_columns + [class_column]

    if config["discretizar"]:
        discretizer = EqualFrequencyDiscretizer(n_bins=config["n_bins"], columns=config["continuous_columns"])
        train_encoded = discretizer.fit_transform(train_df[columns_used])
        test_encoded = discretizer.transform(test_df[columns_used])
    else:
        # Sem discretização: atributos contínuos entram no CBA-RG com seus
        # valores brutos — cada valor numérico distinto vira seu próprio item
        # (dataframe_to_transactions apenas converte cada valor para str).
        train_encoded = train_df[columns_used]
        test_encoded = test_df[columns_used]

    train_transactions = dataframe_to_transactions(train_encoded, class_column)
    test_transactions = dataframe_to_transactions(test_encoded, class_column)
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

    default_usage = _count_default_usage(classifier, test_transactions)

    result = {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "elapsed_seconds": elapsed,
        "num_candidate_rules": len(cars),
        "num_final_rules": len(classifier.rules),
        "avg_antecedent_length": mean(len(rule.antecedent) for rule in classifier.rules)
        if classifier.rules
        else 0.0,
        "accuracy": accuracy,
        "default_class_usage": default_usage,
        "default_class_usage_rate": default_usage / len(test_labels),
    }

    positive_class = config["positive_class"]
    if positive_class is not None:
        true_positives = sum(1 for p, y in zip(predictions, test_labels) if y == positive_class and p == positive_class)
        false_negatives = sum(1 for p, y in zip(predictions, test_labels) if y == positive_class and p != positive_class)
        denom = true_positives + false_negatives
        result["recall_positive_class"] = (true_positives / denom) if denom > 0 else None

    return result


def run_experiment(config: dict) -> dict:
    dataset_path = REPO_ROOT / config["dataset_path"]
    df = pd.read_csv(dataset_path, sep=config["csv_sep"])
    if config.get("id_column") and config["id_column"] in df.columns:
        df = df.drop(columns=[config["id_column"]])

    class_column = config["class_column"]
    if config["feature_columns"] is None:
        config["feature_columns"] = [c for c in df.columns if c != class_column]
    df = df[config["feature_columns"] + [class_column]]

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
    default_usage_rates = [r["default_class_usage_rate"] for r in fold_results]
    recalls = [r["recall_positive_class"] for r in fold_results if r.get("recall_positive_class") is not None]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "discretizar": config["discretizar"],
        "config": config,
        "dependency_versions": dependency_versions(),
        "folds": fold_results,
        "accuracy_mean": mean(accuracies),
        "accuracy_std": pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "num_candidate_rules_mean": mean(r["num_candidate_rules"] for r in fold_results),
        "num_final_rules_mean": mean(r["num_final_rules"] for r in fold_results),
        "avg_antecedent_length_mean": mean(r["avg_antecedent_length"] for r in fold_results),
        "default_class_usage_total": sum(r["default_class_usage"] for r in fold_results),
        "default_class_usage_rate_mean": mean(default_usage_rates),
        "recall_positive_class_mean": mean(recalls) if recalls else None,
        "total_elapsed_seconds": total_elapsed,
    }
    return summary


def write_results(summary: dict, dataset_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    discretizar_suffix = "discretizado" if summary["discretizar"] else "sem_discretizacao"
    detail_path = RESULTS_DIR / f"{dataset_name}_{discretizar_suffix}_{timestamp}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    summary_path = RESULTS_DIR / "resumo.csv"
    # Colunas novas são sempre acrescentadas ao FINAL do cabeçalho e das
    # linhas (nunca inseridas no meio) para preservar a compatibilidade
    # posicional das linhas já gravadas por versões anteriores deste script:
    # o pandas lê linhas mais curtas que o cabeçalho preenchendo NaN à
    # direita, mas quebra se alguma linha tiver mais campos que o cabeçalho.
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
                    "discretizar",
                    "default_class_usage_rate_mean",
                    "recall_positive_class_mean",
                ]
            )
        recall_value = summary["recall_positive_class_mean"]
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
                summary["discretizar"],
                f"{summary['default_class_usage_rate_mean']:.4f}",
                f"{recall_value:.4f}" if recall_value is not None else "",
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
    parser.add_argument(
        "--sem-discretizar",
        action="store_true",
        help="Pula a etapa de discretização, sobrescrevendo 'discretizar' do config para false.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.sem_discretizar:
        config["discretizar"] = False

    summary = run_experiment(config)
    dataset_name = Path(config["dataset_path"]).stem
    detail_path = write_results(summary, dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Discretização: {'sim' if summary['discretizar'] else 'não'}")
    print(f"Seed: {config['seed']}  |  k_folds: {config['k_folds']}")
    print(f"Acurácia média: {summary['accuracy_mean']:.4f} (desvio padrão: {summary['accuracy_std']:.4f})")
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
        print(f"Recall da classe positiva ({config['positive_class']!r}) (média): {summary['recall_positive_class_mean']:.4f}")
    print(f"Tempo total: {summary['total_elapsed_seconds']:.3f}s")
    print(f"Detalhes salvos em: {detail_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
