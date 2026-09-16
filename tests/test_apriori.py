"""Testes do Apriori sobre ``cores_formas_transactions`` (ver conftest.py),
com suportes conferidos manualmente para minsup = 0.25 (>= 2 das 8 linhas):

Pares que passam o suporte mínimo (contagem >= 2):
  {Color=Red, Shape=Circle} = 2/8 = .25    {Color=Red, Label=A} = 3/8 = .375
  {Color=Red, Shape=Square} = 2/8 = .25    {Color=Blue, Label=B} = 3/8 = .375
  {Color=Blue, Shape=Circle} = 2/8 = .25   {Shape=Circle, Label=A} = 2/8 = .25
  {Color=Blue, Shape=Square} = 2/8 = .25   {Shape=Circle, Label=B} = 2/8 = .25
                                           {Shape=Square, Label=A} = 2/8 = .25
                                           {Shape=Square, Label=B} = 2/8 = .25

Pares que falham (contagem == 1, abaixo de .25):
  {Color=Red, Label=B} = 1/8 = .125       {Color=Blue, Label=A} = 1/8 = .125

Isso permite testar tanto o filtro por suporte quanto (no nível 3) a poda
por downward closure: {Color=Red, Shape=Square, Label=A} tem todos os
subconjuntos de tamanho 2 frequentes (então o candidato é gerado), mas
sua própria contagem é 1/8 = .125 -- então é descartado só na checagem de
suporte, não na poda estrutural.
"""

from __future__ import annotations

import pytest

from fase2_implementacao.apriori import apriori_frequent_itemsets


def test_frequent_1_itemsets(cores_formas_transactions):
    frequent, _ = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25)
    for item in [
        ("Color", "Red"),
        ("Color", "Blue"),
        ("Shape", "Circle"),
        ("Shape", "Square"),
        ("Label", "A"),
        ("Label", "B"),
    ]:
        assert frequent[frozenset({item})] == pytest.approx(0.5)


def test_frequent_2_itemsets_pass_minsup(cores_formas_transactions):
    frequent, _ = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25)
    itemset = frozenset({("Color", "Red"), ("Shape", "Circle")})
    assert frequent[itemset] == pytest.approx(0.25)

    itemset = frozenset({("Color", "Red"), ("Label", "A")})
    assert frequent[itemset] == pytest.approx(0.375)


def test_2_itemsets_below_minsup_are_excluded(cores_formas_transactions):
    frequent, _ = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25)
    assert frozenset({("Color", "Red"), ("Label", "B")}) not in frequent
    assert frozenset({("Color", "Blue"), ("Label", "A")}) not in frequent


def test_downward_closure_prunes_3_itemsets_with_infrequent_subset(cores_formas_transactions):
    frequent, _ = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25)

    # Todos os subconjuntos de tamanho 2 são frequentes -> candidato gerado
    # e mantido, pois sua própria contagem (2/8) também passa o minsup.
    kept = frozenset({("Color", "Red"), ("Shape", "Circle"), ("Label", "A")})
    assert frequent[kept] == pytest.approx(0.25)

    # Também tem todos os subconjuntos de tamanho 2 frequentes (candidato
    # gerado), mas sua própria contagem é 1/8 = .125 -> descartado no filtro
    # de suporte, não na poda estrutural.
    dropped = frozenset({("Color", "Red"), ("Shape", "Square"), ("Label", "A")})
    assert dropped not in frequent


def test_max_length_limits_itemset_size(cores_formas_transactions):
    frequent, _ = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25, max_length=1)
    assert all(len(itemset) == 1 for itemset in frequent)


def test_generation_order_is_level_wise(cores_formas_transactions):
    frequent, order = apriori_frequent_itemsets(cores_formas_transactions, min_support=0.25)
    size_1_orders = [order[i] for i in frequent if len(i) == 1]
    size_2_orders = [order[i] for i in frequent if len(i) == 2]
    size_3_orders = [order[i] for i in frequent if len(i) == 3]
    assert max(size_1_orders) < min(size_2_orders)
    assert max(size_2_orders) < min(size_3_orders)

    # Dentro do nível 1, a ordem de geração segue a ordem alfabética
    # canônica dos itens.
    assert order[frozenset({("Color", "Blue")})] < order[frozenset({("Color", "Red")})]
    assert order[frozenset({("Label", "A")})] < order[frozenset({("Label", "B")})]


def test_empty_transactions_and_invalid_min_support_raise():
    with pytest.raises(ValueError):
        apriori_frequent_itemsets([], min_support=0.25)
    with pytest.raises(ValueError):
        apriori_frequent_itemsets([frozenset({("A", "x")})], min_support=1.5)
