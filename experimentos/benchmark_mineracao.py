"""Benchmark controlado das otimizações de mineração (Estágio 5, achado A12).

Compara as implementações *congeladas* anteriores (``tests/_reference``) com
as atuais de Apriori e FP-Growth, sobre as mesmas transações e o mesmo
``min_support``, com repetições alternadas e mediana. Também verifica que os
resultados são idênticos antes de reportar qualquer tempo.

    python -m experimentos.benchmark_mineracao
"""

from __future__ import annotations

import platform
import statistics
import sys
import time

import pandas as pd

from experimentos.runner import REPO_ROOT, load_config
from fase2_implementacao.association_rules import dataframe_to_transactions
from fase2_implementacao.apriori import apriori_frequent_itemsets
from fase2_implementacao.discretization import EqualFrequencyDiscretizer
from fase2_implementacao.fp_growth import fp_growth_frequent_itemsets
from tests._reference.apriori_reference import apriori_frequent_itemsets as ref_apriori
from tests._reference.fp_growth_reference import fp_growth_frequent_itemsets as ref_fp

REPEATS = 5


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def compare(name, old_fn, new_fn):
    old_times, new_times = [], []
    old_result = new_result = None
    for _ in range(REPEATS):  # alternados para dividir igualmente efeitos de cache/CPU
        t, old_result = timed(old_fn)
        old_times.append(t)
        t, new_result = timed(new_fn)
        new_times.append(t)
    old_cmp = old_result[0] if isinstance(old_result, tuple) else old_result
    new_cmp = new_result[0] if isinstance(new_result, tuple) else new_result
    assert old_cmp == new_cmp, f"{name}: resultados diferentes!"
    old_m, new_m = statistics.median(old_times), statistics.median(new_times)
    print(f"{name:<52} antes {old_m*1000:9.1f} ms | depois {new_m*1000:9.1f} ms | x{old_m/new_m:5.2f} | {len(old_cmp)} itemsets")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"Python {platform.python_version()} | {platform.platform()} | mediana de {REPEATS} repetições alternadas")
    for config_name, discretize, min_support, max_len in [
        ("pokemon", True, 0.03, 4),
        ("pokemon", False, 0.03, 4),
        ("iris", True, 0.05, 4),
    ]:
        config = load_config(REPO_ROOT / "experimentos" / "configs" / f"{config_name}.yaml")
        df = pd.read_csv(REPO_ROOT / config["dataset_path"], sep=config["csv_sep"])
        cols = (config["feature_columns"] or [c for c in df.columns if c not in (config["class_column"], config["id_column"])]) + [config["class_column"]]
        data = df[cols]
        if discretize:
            data = EqualFrequencyDiscretizer(config["n_bins"], config["continuous_columns"]).fit_transform(data)
        transactions = dataframe_to_transactions(data, config["class_column"])
        label = f"{config_name} {'discretizado' if discretize else 'bruto'} (N={len(transactions)}, minsup={min_support}, max_len={max_len})"
        compare(f"Apriori   | {label}", lambda: ref_apriori(transactions, min_support, max_len), lambda: apriori_frequent_itemsets(transactions, min_support, max_len))
        compare(f"FP-Growth | {label}", lambda: ref_fp(transactions, min_support, max_len), lambda: fp_growth_frequent_itemsets(transactions, min_support, max_len))


if __name__ == "__main__":
    main()
