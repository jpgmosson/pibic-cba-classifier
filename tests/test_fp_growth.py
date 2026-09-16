"""Testes do FP-Growth: verificação cruzada contra o Apriori.

Apriori e FP-Growth são algoritmos diferentes para o mesmo problema (minerar
itemsets frequentes) -- para qualquer base de dados e minsup, ambos devem
encontrar exatamente o mesmo conjunto de itemsets com os mesmos suportes.
Usamos essa propriedade como teste principal, em vez de recalcular os
suportes manualmente de novo (já feito em test_apriori.py)."""

from __future__ import annotations

import pytest

from fase2_implementacao.apriori import apriori_frequent_itemsets
from fase2_implementacao.fp_growth import fp_growth_frequent_itemsets


def test_matches_apriori_on_cores_formas(cores_formas_transactions):
    apriori_result, _ = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25)
    fp_growth_result = fp_growth_frequent_itemsets(cores_formas_transactions, min_support=0.25)

    assert set(apriori_result) == set(fp_growth_result)
    for itemset, supp in apriori_result.items():
        assert fp_growth_result[itemset] == pytest.approx(supp)


def test_matches_apriori_on_cba_transactions(cba_transactions):
    apriori_result, _ = apriori_frequent_itemsets(cba_transactions, min_support=0.2)
    fp_growth_result = fp_growth_frequent_itemsets(cba_transactions, min_support=0.2)

    assert set(apriori_result) == set(fp_growth_result)
    for itemset, supp in apriori_result.items():
        assert fp_growth_result[itemset] == pytest.approx(supp)


def test_max_length_limits_itemset_size(cores_formas_transactions):
    result = fp_growth_frequent_itemsets(cores_formas_transactions, min_support=0.25, max_length=1)
    assert all(len(itemset) == 1 for itemset in result)


def test_empty_transactions_raise():
    with pytest.raises(ValueError):
        fp_growth_frequent_itemsets([], min_support=0.25)
