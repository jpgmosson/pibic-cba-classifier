"""Testes de aceite da baseline do CBA clássico (Relatório 02: A01, A02, A10).

- A01: corte de erro mínimo por prefixo do M1 (Figura 2 de Liu, Hsu & Ma, 1998).
- A02: um único consequente (o de maior confiança) por antecedente no CBA-RG.
- A10: precedência e limiares com contagens inteiras / frações exatas.
"""

from __future__ import annotations

import random

import pytest

from fase2_implementacao.cba import (
    Classifier,
    Rule,
    build_classifier_m1,
    generate_cars,
    sort_cars_by_precedence,
)

CLS = "Class"


def t(**items):
    return frozenset(items.items())


def rule(ant: dict, label: str, joint: int, ant_count: int, n: int, order: int) -> Rule:
    return Rule(
        antecedent=frozenset(ant.items()),
        consequent=(CLS, label),
        joint_count=joint,
        antecedent_count=ant_count,
        n_transactions=n,
        generation_order=order,
    )


def explicit_training_errors(rules, default, transactions, labels) -> int:
    """Avaliação explícita da lista de decisão no treino (independente do M1)."""
    return sum(Classifier(rules, default, CLS).predict(tr) != y for tr, y in zip(transactions, labels))


# ----------------------------------------------------------------- A01: M1 ---


@pytest.fixture
def relatorio_a01():
    """Exemplo do achado A01: 4x(A=x,B=z,P), 2x(A=y,B=z,N), 1x(A=y,B=z,P)."""
    rows = [("x", "P")] * 4 + [("y", "N")] * 2 + [("y", "P")]
    transactions = [t(A=a, B="z") for a, _ in rows]
    labels = [y for _, y in rows]
    cars = [
        rule({"A": "x"}, "P", 4, 4, 7, 0),  # conf 1
        rule({"B": "z"}, "P", 5, 7, 7, 1),  # conf 5/7
    ]
    return cars, transactions, labels


def test_m1_cut_keeps_only_first_rule_with_default_n(relatorio_a01):
    cars, transactions, labels = relatorio_a01
    clf = build_classifier_m1(cars, transactions, labels, CLS)

    assert [r.antecedent for r in clf.rules] == [frozenset({("A", "x")})]
    assert clf.default_class == "N"
    assert clf.chosen_prefix_length == 1
    # histórico do relatório: (regra 1, padrão N) -> 1 erro; (regras 1+2, padrão P) -> 2 erros
    assert [(r.n_rules, r.rule_errors, r.default_class, r.default_errors, r.total_errors) for r in clf.m1_history] == [
        (1, 0, "N", 1, 1),
        (2, 2, "P", 0, 2),
    ]
    assert explicit_training_errors(clf.rules, clf.default_class, transactions, labels) == 1


def test_m1_history_matches_explicit_evaluation_of_every_prefix(cba_transactions, cba_labels):
    cars = generate_cars(cba_transactions, CLS, min_support=0.2, min_confidence=0.5)
    clf = build_classifier_m1(cars, cba_transactions, cba_labels, CLS)
    selected = [rec.rule for rec in clf.m1_history]
    for rec in clf.m1_history:
        prefix = selected[: rec.n_rules]
        assert rec.total_errors == explicit_training_errors(prefix, rec.default_class, cba_transactions, cba_labels)


def test_m1_history_matches_explicit_evaluation_on_random_data():
    rng = random.Random(7)
    transactions, labels = [], []
    for _ in range(80):
        a, b, c = rng.choice("xyz"), rng.choice("uv"), rng.choice("pq")
        y = rng.choice("NP") if rng.random() < 0.35 else ("P" if (a == "x" or c == "p") else "N")
        transactions.append(t(A=a, B=b, C=c, Class=y))
        labels.append(y)
    cars = generate_cars(transactions, CLS, min_support=0.05, min_confidence=0.5, max_antecedent_length=2)
    clf = build_classifier_m1(cars, transactions, labels, CLS)
    assert clf.m1_history
    selected = [rec.rule for rec in clf.m1_history]
    totals = []
    for rec in clf.m1_history:
        explicit = explicit_training_errors(selected[: rec.n_rules], rec.default_class, transactions, labels)
        assert rec.total_errors == explicit
        totals.append(rec.total_errors)
    # o prefixo escolhido é o PRIMEIRO de menor erro total
    assert clf.chosen_prefix_length == totals.index(min(totals)) + 1
    assert len(clf.rules) == clf.chosen_prefix_length


def test_m1_tie_between_prefixes_keeps_the_shorter_one():
    # Regra 1 (a=1 -> P) acerta 2 e erra 0; regra 2 (b=1 -> N) cobre 1 caso N que o
    # padrão P já erraria: prefixo 2 empata (erro 1) com o prefixo 1 -> fica o prefixo 1.
    transactions = [t(A="1", B="0"), t(A="1", B="0"), t(A="0", B="1"), t(A="0", B="0"), t(A="0", B="0")]
    labels = ["P", "P", "N", "P", "P"]
    cars = [rule({"A": "1"}, "P", 2, 2, 5, 0), rule({"B": "1"}, "N", 1, 1, 5, 1)]
    clf = build_classifier_m1(cars, transactions, labels, CLS)
    totals = [r.total_errors for r in clf.m1_history]
    assert totals == [1, 0]  # sem empate aqui: o prefixo 2 é estritamente melhor
    assert clf.chosen_prefix_length == 2

    # agora força o empate: o caso N é coberto por uma regra que também o erra
    labels2 = ["P", "P", "N", "P", "N"]
    clf2 = build_classifier_m1(cars, transactions, labels2, CLS)
    totals2 = [r.total_errors for r in clf2.m1_history]
    assert totals2[0] == totals2[-1]
    assert clf2.chosen_prefix_length == 1


def test_m1_default_class_comes_from_the_chosen_prefix_not_the_last(relatorio_a01):
    cars, transactions, labels = relatorio_a01
    clf = build_classifier_m1(cars, transactions, labels, CLS)
    assert clf.m1_history[-1].default_class == "P"  # último prefixo
    assert clf.default_class == "N"  # prefixo escolhido


def test_m1_without_cars_returns_training_majority():
    transactions = [t(A="x"), t(A="y"), t(A="z")]
    clf = build_classifier_m1([], transactions, ["P", "N", "N"], CLS)
    assert clf.rules == [] and clf.default_class == "N" and clf.m1_history == []


def test_m1_all_cases_covered_falls_back_to_training_majority():
    transactions = [t(A="x"), t(A="x"), t(A="x")]
    cars = [rule({"A": "x"}, "P", 2, 3, 3, 0)]
    clf = build_classifier_m1(cars, transactions, ["P", "P", "N"], CLS)
    assert [r.n_remaining for r in clf.m1_history] == [0]
    assert clf.default_class == "P"
    assert clf.m1_history[0].total_errors == 1


def test_m1_rejects_empty_training_and_size_mismatch():
    with pytest.raises(ValueError):
        build_classifier_m1([], [], [], CLS)
    with pytest.raises(ValueError):
        build_classifier_m1([], [t(A="x")], ["P", "N"], CLS)


def test_predict_with_rule_reports_which_rule_fired(relatorio_a01):
    cars, transactions, labels = relatorio_a01
    clf = build_classifier_m1(cars, transactions, labels, CLS)
    assert clf.predict_with_rule(t(A="x", B="z")) == ("P", 0)
    assert clf.predict_with_rule(t(A="y", B="z")) == ("N", None)


# ------------------------------------------------- A02: um consequente por antecedente ---


def test_one_consequent_per_antecedent_on_exact_tie():
    transactions = [t(A="x", Class="P")] * 2 + [t(A="x", Class="N")] * 2
    cars = generate_cars(transactions, CLS, min_support=0.1, min_confidence=0.5)
    assert len(cars) == 1
    assert cars[0].class_label == "N"  # empate 2/4: desempate alfabético determinístico
    assert (cars[0].joint_count, cars[0].antecedent_count) == (2, 4)


def test_highest_confidence_consequent_wins_with_three_classes_and_low_minconf():
    transactions = [t(A="x", Class="C")] * 3 + [t(A="x", Class="B")] * 2 + [t(A="x", Class="A")] * 1
    cars = generate_cars(transactions, CLS, min_support=0.1, min_confidence=0.3)
    assert [(c.class_label, c.joint_count, c.antecedent_count) for c in cars] == [("C", 3, 6)]


def test_winner_below_minconf_leaves_no_rule_for_that_antecedent():
    transactions = [t(A="x", Class="C")] * 2 + [t(A="x", Class="B")] * 2 + [t(A="x", Class="A")] * 2
    assert generate_cars(transactions, CLS, min_support=0.1, min_confidence=0.5) == []


def test_generate_cars_has_unique_antecedents_and_is_deterministic(cba_transactions):
    first = generate_cars(cba_transactions, CLS, min_support=0.1, min_confidence=0.3)
    second = generate_cars(cba_transactions, CLS, min_support=0.1, min_confidence=0.3)
    assert first == second
    assert len({c.antecedent for c in first}) == len(first)
    assert [c.generation_order for c in first] == sorted(c.generation_order for c in first)


# ------------------------------------------------------ A10: comparação exata ---


def test_precedence_uses_exact_ratios_not_floats():
    # (2/10)/(6/10) e (3/10)/(9/10) são ambos 1/3, mas diferem em ponto flutuante.
    assert (2 / 10) / (6 / 10) != (3 / 10) / (9 / 10)
    r_small = rule({"A": "x"}, "P", 2, 6, 10, 0)
    r_big = rule({"A": "y"}, "P", 3, 9, 10, 1)
    assert r_small.confidence_fraction == r_big.confidence_fraction
    assert [r.antecedent for r in sort_cars_by_precedence([r_small, r_big])] == [r_big.antecedent, r_small.antecedent]


def test_precedence_final_tie_break_is_generation_order():
    a = rule({"A": "x"}, "P", 2, 4, 10, 5)
    b = rule({"A": "y"}, "P", 2, 4, 10, 2)
    assert sort_cars_by_precedence([a, b]) == [b, a]


def test_minconf_and_minsup_boundaries_are_exact():
    # sup(A=x,P)=3/10, conf=3/5
    transactions = [t(A="x", Class="P")] * 3 + [t(A="x", Class="N")] * 2 + [t(A="y", Class="N")] * 5
    assert any(c.class_label == "P" for c in generate_cars(transactions, CLS, 0.3, 0.6))  # exatamente na fronteira
    assert not any(c.class_label == "P" for c in generate_cars(transactions, CLS, 0.3, 0.61))  # logo acima
    assert any(c.class_label == "P" for c in generate_cars(transactions, CLS, 0.3, 0.59))  # logo abaixo
    assert not any(c.class_label == "P" for c in generate_cars(transactions, CLS, 0.31, 0.5))  # minsup acima


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_support": 0},
        {"min_support": float("nan")},
        {"min_support": 1.5},
        {"min_support": True},
        {"min_confidence": float("nan")},
        {"min_confidence": -0.1},
        {"min_confidence": 1.1},
        {"max_antecedent_length": 0},
        {"max_antecedent_length": True},
    ],
)
def test_generate_cars_rejects_invalid_parameters(cba_transactions, kwargs):
    params = {"min_support": 0.2, "min_confidence": 0.5, **kwargs}
    with pytest.raises(ValueError):
        generate_cars(cba_transactions, CLS, **params)
