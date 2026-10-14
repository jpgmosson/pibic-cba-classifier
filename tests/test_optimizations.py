"""Regressão das otimizações do Estágio 5 (achado A12 do Relatório 02).

Uma otimização não pode mudar itemsets, suportes, ordem de geração nem CARs.
Os oráculos são (a) cópias congeladas das implementações anteriores
(``tests/_reference``) e (b) enumeração exaustiva independente.
"""

from __future__ import annotations

import itertools
import random

import pytest

from fase2_implementacao.apriori import apriori_frequent_itemset_counts, apriori_frequent_itemsets
from fase2_implementacao.fp_growth import fp_growth_frequent_itemsets
from tests._reference.apriori_reference import apriori_frequent_itemsets as reference_apriori
from tests._reference.fp_growth_reference import fp_growth_frequent_itemsets as reference_fp_growth


def random_transactions(seed: int):
    rng = random.Random(seed)
    n_attrs = rng.randint(2, 5)
    n_values = rng.randint(2, 4)
    n = rng.randint(5, 40)
    return [
        frozenset((f"A{a}", rng.randrange(n_values)) for a in range(n_attrs) if rng.random() < 0.85) | {("Class", rng.choice("PN"))}
        for _ in range(n)
    ]


def brute_force(transactions, min_count):
    items = sorted({item for t in transactions for item in t})
    result = {}
    for size in range(1, len(items) + 1):
        found = False
        for combo in itertools.combinations(items, size):
            itemset = frozenset(combo)
            count = sum(itemset <= t for t in transactions)
            if count >= min_count:
                result[itemset] = count
                found = True
        if not found:
            break
    return result


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("min_support", [0.1, 0.25])
def test_apriori_matches_reference_exactly_including_generation_order(seed, min_support):
    transactions = random_transactions(seed)
    new_supports, new_order = apriori_frequent_itemsets(transactions, min_support)
    ref_supports, ref_order = reference_apriori(transactions, min_support)
    assert new_supports == ref_supports
    assert new_order == ref_order
    assert list(new_supports) == list(ref_supports)  # mesma ordem de inserção


@pytest.mark.parametrize("seed", range(40))
def test_apriori_counts_match_exhaustive_enumeration(seed):
    transactions = random_transactions(seed)
    counts, _, n = apriori_frequent_itemset_counts(transactions, 0.2)
    min_count = -(-n * 2 // 10)  # ceil(0.2 * n)
    assert counts == brute_force(transactions, min_count)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("max_length", [None, 2])
def test_fp_growth_matches_reference_and_apriori(seed, max_length):
    transactions = random_transactions(seed)
    new = fp_growth_frequent_itemsets(transactions, 0.15, max_length)
    assert new == reference_fp_growth(transactions, 0.15, max_length)
    apriori, _ = apriori_frequent_itemsets(transactions, 0.15, max_length)
    assert new == apriori


def test_apriori_skips_attribute_conflicts_without_changing_results():
    # Itens do mesmo atributo nunca coexistem; o resultado é idêntico ao da contagem ingênua.
    transactions = [frozenset({("A", 1), ("B", 1), ("Class", "P")})] * 3 + [frozenset({("A", 2), ("B", 1), ("Class", "N")})] * 3
    new, _ = apriori_frequent_itemsets(transactions, 0.3)
    ref, _ = reference_apriori(transactions, 0.3)
    assert new == ref
    assert frozenset({("A", 1), ("A", 2)}) not in new
