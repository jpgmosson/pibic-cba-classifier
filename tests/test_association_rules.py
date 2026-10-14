"""Testes de suporte, confiança e lift, com valores conferidos manualmente
sobre o dataset ``cores_formas_transactions`` (ver conftest.py).

Suportes de referência (8 transações):
  sup(Color=Red) = 4/8 = 0.5          sup(Color=Blue) = 4/8 = 0.5
  sup(Shape=Circle) = 4/8 = 0.5       sup(Shape=Square) = 4/8 = 0.5
  sup(Label=A) = 4/8 = 0.5            sup(Label=B) = 4/8 = 0.5
  sup(Color=Red, Label=A) = 3/8 = 0.375   (linhas 1, 2, 3)
  sup(Color=Blue, Label=B) = 3/8 = 0.375  (linhas 5, 6, 7)

conf(Color=Red => Label=A) = 0.375 / 0.5 = 0.75
lift(Color=Red => Label=A) = 0.75 / 0.5 = 1.5
"""

from __future__ import annotations

import pytest

from fase2_implementacao.association_rules import confidence, lift, support


def test_support_of_single_item(cores_formas_transactions):
    assert support({("Color", "Red")}, cores_formas_transactions) == pytest.approx(0.5)
    assert support({("Shape", "Circle")}, cores_formas_transactions) == pytest.approx(0.5)
    assert support({("Label", "A")}, cores_formas_transactions) == pytest.approx(0.5)


def test_support_of_empty_itemset_is_one(cores_formas_transactions):
    assert support(frozenset(), cores_formas_transactions) == 1.0


def test_support_requires_transactions():
    with pytest.raises(ValueError):
        support({("Color", "Red")}, [])


def test_support_of_conjoined_itemset(cores_formas_transactions):
    itemset = {("Color", "Red"), ("Label", "A")}
    assert support(itemset, cores_formas_transactions) == pytest.approx(3 / 8)


def test_confidence_matches_hand_calculation(cores_formas_transactions):
    conf = confidence({("Color", "Red")}, {("Label", "A")}, cores_formas_transactions)
    assert conf == pytest.approx(0.75)

    conf_blue = confidence({("Color", "Blue")}, {("Label", "B")}, cores_formas_transactions)
    assert conf_blue == pytest.approx(0.75)


def test_confidence_with_unsupported_antecedent_is_zero(cores_formas_transactions):
    conf = confidence({("Color", "Green")}, {("Label", "A")}, cores_formas_transactions)
    assert conf == 0.0


def test_lift_matches_hand_calculation(cores_formas_transactions):
    value = lift({("Color", "Red")}, {("Label", "A")}, cores_formas_transactions)
    assert value == pytest.approx(1.5)


def test_lift_of_independent_attributes_is_one(cores_formas_transactions):
    # Shape=Circle e Shape=Square têm exatamente metade de Label=A cada
    # (2 em 4), igual à proporção geral de Label=A (4/8) -> lift == 1.
    value = lift({("Shape", "Circle")}, {("Label", "A")}, cores_formas_transactions)
    assert value == pytest.approx(1.0)


# ------------------------------------------------------------ inferência sem alvo / ausências ---

import pandas as pd  # noqa: E402

from fase2_implementacao.association_rules import dataframe_to_feature_transactions, dataframe_to_transactions  # noqa: E402


def test_feature_transactions_exclude_the_class_column():
    df = pd.DataFrame({"A": ["x", "y"], "Class": ["P", "N"]})
    assert dataframe_to_feature_transactions(df, "Class") == [frozenset({("A", "x")}), frozenset({("A", "y")})]
    assert dataframe_to_feature_transactions(df[["A"]]) == [frozenset({("A", "x")}), frozenset({("A", "y")})]
    assert frozenset({("Class", "P"), ("A", "x")}) == dataframe_to_transactions(df, "Class")[0]


def test_missing_values_are_rejected_instead_of_becoming_the_string_nan():
    df = pd.DataFrame({"A": ["x", None], "Class": ["P", "N"]})
    with pytest.raises(ValueError, match="ausentes"):
        dataframe_to_transactions(df, "Class")
    with pytest.raises(ValueError, match="ausentes"):
        dataframe_to_feature_transactions(df, "Class")


def test_mixed_int_and_float_columns_encode_identically_with_and_without_the_class_column():
    # Antes da correção: treino (com a coluna de classe, dtype object) gerava "45" e
    # o teste (só preditores, int+float promovidos a float) gerava "45.0".
    df = pd.DataFrame({"i": [45, 60], "f": [0.5, 1.5], "Class": ["P", "N"]})
    train = dataframe_to_transactions(df, "Class")
    test = dataframe_to_feature_transactions(df, "Class")
    assert all(item in train[row] for row in range(2) for item in test[row])
    assert ("i", "45") in test[0] and ("f", "0.5") in test[0]
