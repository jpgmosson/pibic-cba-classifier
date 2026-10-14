"""Testes de CBA-RG e CBA-CB (M1) sobre ``cba_transactions`` (ver
conftest.py), com minsup=0.2 e minconf=0.5.

Cálculo manual (10 transações, atributo A ∈ {x, y, z, w}, classe ∈ {P, N}):

  sup(A=x)=.3  sup(A=y)=.3  sup(A=z)=.2  sup(A=w)=.2
  sup(A=x,P)=.2 conf=.667 (2/3)   -> CAR: A=x => P
  sup(A=x,N)=.1 -> abaixo do minsup, descartada antes mesmo de checar confiança
  sup(A=y,N)=.2 conf=.667 (2/3)   -> CAR: A=y => N
  sup(A=y,P)=.1 -> abaixo do minsup
  sup(A=z,N)=.2 conf=1.0          -> CAR: A=z => N
  sup(A=w,P)=.1 e sup(A=w,N)=.1   -> ambas abaixo do minsup: nenhuma CAR
                                     gerada para A=w

3 CARs no total. Ordenadas por precedência (confiança desc, suporte desc,
ordem de geração asc):
  1º) A=z => N   (confiança 1.0)
  2º) A=x => P   (confiança .667, gerada antes de A=y=>N na varredura do
                  Apriori -- ambas têm confiança e suporte idênticos)
  3º) A=y => N   (confiança .667)

Cobertura M1 (as 3 regras são selecionadas por cobertura) e histórico de erro
por prefixo (Figura 2 de Liu, Hsu & Ma, 1998):

  prefixo 1 = [z=>N]           erros das regras 0; residual = x(2P,1N)+y(2N,1P)
                               +w(1P,1N) = 4P/4N -> padrão N (empate, alfabético),
                               4 erros              -> total 4
  prefixo 2 = [z=>N, x=>P]     erros das regras 1; residual = y(2N,1P)+w(1P,1N)
                               = 3N/2P -> padrão N, 2 erros -> total 3
  prefixo 3 = [.., y=>N]       erros das regras 2; residual = w(1P,1N)
                               -> padrão N, 1 erro  -> total 3

O primeiro mínimo é o prefixo 2 (empata com o 3, que é descartado): o
classificador final é [z=>N, x=>P] com padrão N.
"""

from __future__ import annotations

import pytest

from fase2_implementacao.cba import (
    build_classifier_m1,
    generate_cars,
    majority_class,
    sort_cars_by_precedence,
)


def test_generate_cars_respects_minsup_and_minconf(cba_transactions):
    cars = generate_cars(cba_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    rules = {(r.antecedent, r.consequent): r for r in cars}

    assert len(cars) == 3
    x_to_p = rules[(frozenset({("A", "x")}), ("Class", "P"))]
    assert x_to_p.support == pytest.approx(0.2)
    assert x_to_p.confidence == pytest.approx(2 / 3)

    y_to_n = rules[(frozenset({("A", "y")}), ("Class", "N"))]
    assert y_to_n.confidence == pytest.approx(2 / 3)

    z_to_n = rules[(frozenset({("A", "z")}), ("Class", "N"))]
    assert z_to_n.confidence == pytest.approx(1.0)

    # A=w nunca passa no minsup em nenhuma classe -> nenhuma CAR para A=w.
    assert not any(r.antecedent == frozenset({("A", "w")}) for r in cars)


def test_generate_cars_respects_max_antecedent_length(cba_transactions):
    cars = generate_cars(
        cba_transactions,
        class_attribute="Class",
        min_support=0.2,
        min_confidence=0.5,
        max_antecedent_length=1,
    )
    assert all(len(rule.antecedent) <= 1 for rule in cars)


def test_sort_cars_by_precedence_breaks_ties_by_generation_order(cba_transactions):
    cars = generate_cars(cba_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    ordered = sort_cars_by_precedence(cars)

    assert [(r.antecedent, r.class_label) for r in ordered] == [
        (frozenset({("A", "z")}), "N"),
        (frozenset({("A", "x")}), "P"),
        (frozenset({("A", "y")}), "N"),
    ]


def test_majority_class_breaks_ties_alphabetically():
    assert majority_class(["P", "N"]) == "N"
    assert majority_class(["N", "N", "P"]) == "N"
    assert majority_class(["P", "P", "N"]) == "P"


def test_build_classifier_m1_end_to_end(cba_transactions, cba_labels):
    cars = generate_cars(cba_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    classifier = build_classifier_m1(cars, cba_transactions, cba_labels, class_attribute="Class")

    # Corte de erro mínimo do M1: A=y=>N (prefixo 3, erro total 3) empata com o
    # prefixo 2 e é descartado -- o primeiro mínimo vence.
    assert [(r.antecedent, r.class_label) for r in classifier.rules] == [
        (frozenset({("A", "z")}), "N"),
        (frozenset({("A", "x")}), "P"),
    ]
    assert classifier.default_class == "N"
    assert [rec.total_errors for rec in classifier.m1_history] == [4, 3, 3]
    assert classifier.chosen_prefix_length == 2


def test_classifier_predict_uses_first_matching_rule_then_default(cba_transactions):
    cars = generate_cars(cba_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    labels = [dict(t)["Class"] for t in cba_transactions]
    classifier = build_classifier_m1(cars, cba_transactions, labels, class_attribute="Class")

    assert classifier.predict(frozenset({("A", "z"), ("Class", "?")})) == "N"
    assert classifier.predict(frozenset({("A", "x"), ("Class", "?")})) == "P"
    assert classifier.predict(frozenset({("A", "y"), ("Class", "?")})) == "N"
    # A=w não bate com nenhuma regra -> classe padrão.
    assert classifier.predict(frozenset({("A", "w"), ("Class", "?")})) == "N"
