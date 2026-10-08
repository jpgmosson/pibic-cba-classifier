"""Testes do CBA-QS (fase2_implementacao/cba_qs.py), incluindo um cenário
desenhado à mão em que ele deve inverter a precedência do CBA-CB M1
clássico: uma regra de confiança mais alta mas "óbvia" para a classe
majoritária deve perder para uma regra de confiança mais baixa, porém muito
mais interessante (fator de certeza e lift bem acima do prior), para a
classe minoritária.

Cálculo manual do cenário (10 transações, atributo A ∈ {p, q, r}, classe ∈
{Maj, Min}):

  A=p: 5 transações (4 Maj, 1 Min) -> sup(A=p)=.5
  A=q: 4 transações (1 Maj, 3 Min) -> sup(A=q)=.4
  A=r: 1 transação  (1 Maj)        -> sup(A=r)=.1 (abaixo do minsup=.2, não
                                       gera nenhum itemset/CAR)

  prior(Maj) = 6/10 = .6   prior(Min) = 4/10 = .4

  CAR 1: A=p => Maj
    sup(A=p,Maj) = .4      conf = .4/.5 = .8
    CF   = (.8-.6)/(1-.6) = .5
    lift = .8/.6           = 1.333...

  CAR 2: A=q => Min
    sup(A=q,Min) = .3      conf = .3/.4 = .75
    CF   = (.75-.4)/(1-.4) = .5833...
    lift = .75/.4           = 1.875

  sup(A=p,Min)=.1 e sup(A=q,Maj)=.1 ficam abaixo do minsup=.2 -> nenhuma
  outra CAR é gerada. Exatamente 2 CARs candidatas.

  M1 (confiança desc): CAR 1 (.8) > CAR 2 (.75) -> ordem [CAR1, CAR2].

  CBA-QS, pesos iguais (.25 cada), normalização min-max entre as 2 CARs:
    Q (CF):      CAR1=0.0  CAR2=1.0   (min=.5, max=.5833)
    I (lift):    CAR1=0.0  CAR2=1.0   (min=1.333, max=1.875)
    C (sup(X)):  CAR1=1.0  CAR2=0.0   (min=.4 [CAR2], max=.5 [CAR1])
    L (tamanho): ambos len=1 -> faixa zero -> 0.5 para os dois

    S(CAR1) = .25*0 + .25*0 + .25*1.0 - .25*.5 = .125
    S(CAR2) = .25*1.0 + .25*1.0 + .25*0 - .25*.5 = .375

  S(CAR2) > S(CAR1) -> CBA-QS inverte a ordem para [CAR2, CAR1]: a regra da
  classe minoritária (A=q=>Min) passa a ter precedência sobre a regra da
  classe majoritária (A=p=>Maj), apesar de confiança e suporte menores.
"""

from __future__ import annotations

import pytest

from fase2_implementacao.cba import build_classifier_m1, generate_cars, sort_cars_by_precedence
from fase2_implementacao.cba_qs import (
    build_classifier_cba_qs,
    certainty_factor,
    score_cars,
    sort_cars_by_score,
)


def _transaction(**items):
    return frozenset(items.items())


@pytest.fixture
def imbalanced_transactions():
    rows = [
        ("p", "Maj"),
        ("p", "Maj"),
        ("p", "Maj"),
        ("p", "Maj"),
        ("p", "Min"),
        ("q", "Min"),
        ("q", "Min"),
        ("q", "Min"),
        ("q", "Maj"),
        ("r", "Maj"),
    ]
    return [_transaction(A=a, Class=label) for a, label in rows]


@pytest.fixture
def imbalanced_labels(imbalanced_transactions):
    return [dict(t)["Class"] for t in imbalanced_transactions]


def test_certainty_factor_hand_computed_values():
    # Regra melhor que o acaso (conf >= prior): normaliza pela distância até 1.0.
    assert certainty_factor(confidence=0.8, prior=0.6) == pytest.approx(0.5)
    # Regra pior que o acaso (conf < prior): normaliza pela distância até 0.0.
    assert certainty_factor(confidence=0.3, prior=0.6) == pytest.approx(-0.5)
    # Regra exatamente no prior: nem melhor nem pior que o acaso.
    assert certainty_factor(confidence=0.6, prior=0.6) == pytest.approx(0.0)


def test_generate_cars_produces_exactly_the_two_expected_cars(imbalanced_transactions):
    cars = generate_cars(imbalanced_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    rules = {(r.antecedent, r.consequent): r for r in cars}

    assert len(cars) == 2
    car1 = rules[(frozenset({("A", "p")}), ("Class", "Maj"))]
    assert car1.confidence == pytest.approx(0.8)
    assert car1.support == pytest.approx(0.4)

    car2 = rules[(frozenset({("A", "q")}), ("Class", "Min"))]
    assert car2.confidence == pytest.approx(0.75)
    assert car2.support == pytest.approx(0.3)


def test_m1_ranks_the_majority_rule_first(imbalanced_transactions):
    cars = generate_cars(imbalanced_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    ordered = sort_cars_by_precedence(cars)

    assert [(r.antecedent, r.class_label) for r in ordered] == [
        (frozenset({("A", "p")}), "Maj"),
        (frozenset({("A", "q")}), "Min"),
    ]


def test_cba_qs_score_matches_hand_calculation(imbalanced_transactions):
    cars = generate_cars(imbalanced_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    scored = {(sr.rule.antecedent, sr.rule.class_label): sr.score for sr in score_cars(cars, imbalanced_transactions)}

    assert scored[(frozenset({("A", "p")}), "Maj")] == pytest.approx(0.125)
    assert scored[(frozenset({("A", "q")}), "Min")] == pytest.approx(0.375)


def test_cba_qs_inverts_precedence_to_favor_the_minority_rule(imbalanced_transactions):
    cars = generate_cars(imbalanced_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)
    ordered = sort_cars_by_score(cars, imbalanced_transactions)

    # Ordem invertida em relação ao M1: a regra da classe minoritária,
    # muito mais "interessante" (CF e lift bem acima do prior), agora vem
    # primeiro, apesar de confiança e suporte menores.
    assert [(r.antecedent, r.class_label) for r in ordered] == [
        (frozenset({("A", "q")}), "Min"),
        (frozenset({("A", "p")}), "Maj"),
    ]


def test_build_classifier_cba_qs_selects_minority_rule_first(imbalanced_transactions, imbalanced_labels):
    cars = generate_cars(imbalanced_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)

    m1_classifier = build_classifier_m1(cars, imbalanced_transactions, imbalanced_labels, class_attribute="Class")
    qs_classifier = build_classifier_cba_qs(cars, imbalanced_transactions, imbalanced_labels, class_attribute="Class")

    assert m1_classifier.rules[0].class_label == "Maj"
    assert qs_classifier.rules[0].class_label == "Min"
    # A única transação com A=r nunca é coberta por nenhuma CAR -> classe
    # padrão = majoritária entre os não cobertos = Maj (é a única restante),
    # igual nos dois classificadores, já que a cobertura em si (quais
    # transações cada regra cobre) não muda com a ordem aqui (os valores de
    # A usados pelas duas CARs são disjuntos entre si).
    assert m1_classifier.default_class == "Maj"
    assert qs_classifier.default_class == "Maj"


def test_cba_qs_weights_are_configurable(imbalanced_transactions):
    cars = generate_cars(imbalanced_transactions, class_attribute="Class", min_support=0.2, min_confidence=0.5)

    # Com peso zero em Q e I (e todo o peso em C, cobertura), a regra da
    # classe majoritária (maior sup(antecedente)) volta a vencer.
    ordered = sort_cars_by_score(cars, imbalanced_transactions, weights=(0.0, 0.0, 1.0, 0.0))
    assert ordered[0].class_label == "Maj"
