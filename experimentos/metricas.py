"""Métricas de classificação calculadas a partir de predições individuais.

Tudo aqui depende só de ``(y_true, y_pred)``, de modo que as métricas de uma
execução podem ser **recalculadas a partir do arquivo de predições fora da
dobra (OOF)** salvo pelo runner e comparadas com o resumo gravado (achado
A11 do Relatório 02):

    python -m experimentos.metricas --oof experimentos/resultados/<run_id>/oof_predictions.csv

Implementação própria, sem scikit-learn. Definições:

- ``accuracy``: acertos / N.
- ``balanced_accuracy``: média do recall de cada classe presente em ``y_true``.
- ``macro_f1``: média simples do F1 por classe (F1 = 0 quando precisão e
  recall são ambos 0 ou indefinidos), sobre todas as classes de ``labels``.
- ``mcc``: coeficiente de correlação de Matthews multiclasse (Gorodkin);
  0 quando o denominador é nulo (ex.: classificador que prevê uma classe só).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Sequence
from pathlib import Path


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[list[int]]:
    """Matriz ``m[i][j]`` = nº de casos da classe real ``labels[i]`` previstos como ``labels[j]``."""
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return matrix


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho")
    if len(y_true) == 0:
        raise ValueError("não há predições para avaliar")
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels)
    n = len(y_true)
    k = len(labels)

    row_sums = [sum(row) for row in matrix]  # real por classe (t_k)
    col_sums = [sum(matrix[i][j] for i in range(k)) for j in range(k)]  # previsto por classe (p_k)
    correct = sum(matrix[i][i] for i in range(k))

    per_class = {}
    recalls_present = []
    f1s = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        precision = tp / col_sums[i] if col_sums[i] else None
        recall = tp / row_sums[i] if row_sums[i] else None
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": row_sums[i]}
        f1s.append(f1)
        if row_sums[i]:
            recalls_present.append(recall)

    numerator = correct * n - sum(p * t for p, t in zip(col_sums, row_sums))
    denominator = math.sqrt((n * n - sum(p * p for p in col_sums)) * (n * n - sum(t * t for t in row_sums)))
    mcc = numerator / denominator if denominator else 0.0

    return {
        "n": n,
        "accuracy": correct / n,
        "balanced_accuracy": sum(recalls_present) / len(recalls_present),
        "macro_f1": sum(f1s) / len(f1s),
        "mcc": mcc,
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "rows": "classe real", "columns": "classe prevista", "matrix": matrix},
    }


METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "mcc")


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    units: Sequence,
    labels: Sequence[str],
    reps: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, list[float]]:
    """Intervalo de confiança percentílico por bootstrap das métricas OOF.

    Mede a **incerteza de generalização** (quanto a métrica agregada mudaria
    com outra amostra de casos), distinta da *dispersão entre folds*. A
    reamostragem é por ``units``: passe os grupos (ex.: espécie) quando
    existirem, para não tratar como independentes linhas da mesma espécie, ou
    os índices das linhas caso contrário.
    """
    members: dict = {}
    for i, u in enumerate(units):
        members.setdefault(u, []).append(i)
    keys = list(members)
    rng = random.Random(seed)
    samples = {k: [] for k in METRIC_KEYS}
    for _ in range(reps):
        idx = [i for u in rng.choices(keys, k=len(keys)) for i in members[u]]
        m = compute_metrics([y_true[i] for i in idx], [y_pred[i] for i in idx], labels=labels)
        for k in METRIC_KEYS:
            samples[k].append(m[k])
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for k, values in samples.items():
        values.sort()
        out[k] = [values[int(lo * (reps - 1))], values[int(hi * (reps - 1))]]
    return out


def read_oof(path: Path) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["y_true"] for r in rows], [r["y_pred"] for r in rows]


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Recalcula métricas a partir de um oof_predictions.csv.")
    parser.add_argument("--oof", required=True, type=Path)
    args = parser.parse_args(argv)
    y_true, y_pred = read_oof(args.oof)
    print(json.dumps(compute_metrics(y_true, y_pred), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
