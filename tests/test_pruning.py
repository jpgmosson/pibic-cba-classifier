"""Testes da taxa de erro pessimista e da poda pessimista de condições (CBA-RG).

Valores à mão (nível de confiança 0,25):

* ``U(0, n) = 1 - 0,25**(1/n)``:  U(0,5) = 0,2421;  U(0,20) = 0,0670;  U(0,15) = 0,0883.
* Cenário de poda: 20 casos ``A=x`` (todos ``P``), dos quais 5 com ``B=u`` e 15 com ``B=v``,
  e 20 casos ``A=y,B=v`` (todos ``N``). A regra ``A=x,B=u => P`` (0 erros em 5 casos, 0,2421)
  é pior que ``A=x => P`` (0 erros em 20, 0,0670) e é podada; o mesmo para ``A=x,B=v => P``
  (0 erros em 15, 0,0883). Já ``A=x => P`` e ``B=u => P`` (0 erros em 5, 0,2421, contra a
  regra vazia ``=> P`` com 20 erros em 40) permanecem.
"""

from __future__ import annotations

import math

import pytest

from fase2_implementacao.cba import generate_cars, prune_cars_pessimistic
from fase2_implementacao.pruning import pessimistic_error_rate


def t(**items):
    return frozenset(items.items())


def binomial_cdf(e, n, p):  # independente do código: usa math.comb exato
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(e + 1))


# --------------------------------------------------------------- taxa pessimista ---


@pytest.mark.parametrize("n", [1, 5, 15, 20, 100])
def test_zero_errors_matches_closed_form(n):
    assert pessimistic_error_rate(0, n, 0.25) == pytest.approx(1 - 0.25 ** (1 / n), abs=1e-12)


def test_hand_values_for_zero_errors():
    assert pessimistic_error_rate(0, 5) == pytest.approx(0.2421, abs=1e-4)
    assert pessimistic_error_rate(0, 20) == pytest.approx(0.0670, abs=1e-4)
    assert pessimistic_error_rate(0, 15) == pytest.approx(0.0883, abs=1e-4)


@pytest.mark.parametrize("e,n", [(1, 10), (2, 10), (5, 10), (3, 40), (10, 100), (30, 200)])
@pytest.mark.parametrize("cf", [0.1, 0.25, 0.5])
def test_rate_is_the_binomial_upper_limit(e, n, cf):
    p = pessimistic_error_rate(e, n, cf)
    assert binomial_cdf(e, n, p) == pytest.approx(cf, abs=1e-9)  # P(Bin(n,p) <= e) = cf
    assert e / n < p < 1


def test_all_wrong_gives_one_and_monotonicity():
    assert pessimistic_error_rate(7, 7) == 1.0
    rates = [pessimistic_error_rate(e, 20) for e in range(0, 21)]
    assert rates == sorted(rates) and len(set(rates)) == len(rates)  # cresce com os erros
    by_n = [pessimistic_error_rate(0, n) for n in (1, 2, 5, 10, 50)]
    assert by_n == sorted(by_n, reverse=True)  # mais casos, menos pessimista
    assert pessimistic_error_rate(2, 20, 0.1) > pessimistic_error_rate(2, 20, 0.5)  # menor cf = mais pessimista


@pytest.mark.parametrize("args", [(-1, 5), (6, 5), (0, 0), (1.5, 5), (True, 5)])
def test_invalid_counts_are_rejected(args):
    with pytest.raises(ValueError):
        pessimistic_error_rate(*args)


@pytest.mark.parametrize("cf", [0, 1, -0.1, 1.5, float("nan"), True])
def test_invalid_cf_is_rejected(cf):
    with pytest.raises(ValueError):
        pessimistic_error_rate(1, 10, cf)


# ----------------------------------------------------------------- poda de regras ---


@pytest.fixture
def cenario():
    return (
        [t(A="x", B="u", Class="P")] * 5
        + [t(A="x", B="v", Class="P")] * 15
        + [t(A="y", B="v", Class="N")] * 20
    )


def keys(cars):
    return {(tuple(sorted(c.antecedent)), c.class_label) for c in cars}


def test_pruning_removes_longer_rules_that_are_worse_than_a_shortened_version(cenario):
    plain = generate_cars(cenario, "Class", 0.05, 0.5)
    pruned = generate_cars(cenario, "Class", 0.05, 0.5, prune=True)
    ax_u = (tuple(sorted({("A", "x"), ("B", "u")})), "P")
    ax_v = (tuple(sorted({("A", "x"), ("B", "v")})), "P")
    assert ax_u in keys(plain) and ax_v in keys(plain)
    assert ax_u not in keys(pruned) and ax_v not in keys(pruned)
    assert ((("A", "x"),), "P") in keys(pruned)
    assert ((("B", "u"),), "P") in keys(pruned)  # 0,2421 < regra vazia (20 erros em 40)


def test_pruned_cars_are_an_ordered_subsequence_of_the_unpruned(cenario):
    plain = generate_cars(cenario, "Class", 0.05, 0.5)
    pruned = generate_cars(cenario, "Class", 0.05, 0.5, prune=True)
    it = iter(plain)
    assert all(any(c is p or c == p for p in it) for c in pruned)  # subsequência estrita
    assert 0 < len(pruned) < len(plain)


def test_prune_is_off_by_default_and_generation_is_unchanged(cenario):
    assert generate_cars(cenario, "Class", 0.05, 0.5) == generate_cars(cenario, "Class", 0.05, 0.5, prune=False)


def test_longer_rule_is_kept_when_it_is_better_than_its_shortened_version():
    data = [t(A="x", B="u", Class="P")] * 5 + [t(A="x", B="v", Class="N")] * 5 + [t(A="y", B="v", Class="N")] * 10
    pruned = generate_cars(data, "Class", 0.05, 0.5, prune=True)
    # (A=x,B=u => P): 0 erros em 5 (0,2421);  A=x => P/N: 5 erros em 10 (~0,7)
    assert ((("A", "x"), ("B", "u")), "P") in keys(pruned)


def test_single_condition_rule_weaker_than_the_class_prior_is_pruned():
    data = [t(C="z", Class="P")] * 8 + [t(C="z", Class="N")] * 2 + [t(C="w", Class="P")] * 82 + [t(C="w", Class="N")] * 8
    # a regra C=z => P (2 erros em 10) é pior que prever P para todos (10 erros em 100)
    assert pessimistic_error_rate(2, 10) > pessimistic_error_rate(10, 100)
    plain, pruned = generate_cars(data, "Class", 0.05, 0.5), generate_cars(data, "Class", 0.05, 0.5, prune=True)
    assert ((("C", "z"),), "P") in keys(plain) and ((("C", "z"),), "P") not in keys(pruned)
    assert pessimistic_error_rate(8, 90) < pessimistic_error_rate(10, 100)
    assert ((("C", "w"),), "P") in keys(pruned)


def test_ties_do_not_prune_so_redundant_conditions_with_equal_counts_are_kept():
    # A=x e B=u ocorrem sempre juntos: (A=x,B=u => P) tem as mesmas contagens que (A=x => P)
    data = [t(A="x", B="u", Class="P")] * 10 + [t(A="y", B="v", Class="N")] * 10
    pruned = generate_cars(data, "Class", 0.05, 0.5, prune=True)
    assert ((("A", "x"), ("B", "u")), "P") in keys(pruned)  # "estritamente maior": empate não poda


def test_prune_cars_pessimistic_is_a_pure_function_of_counts(cenario):
    from fase2_implementacao.apriori import apriori_frequent_itemset_counts

    counts, _, n = apriori_frequent_itemset_counts(cenario, 0.05, None)
    plain = generate_cars(cenario, "Class", 0.05, 0.5)
    assert prune_cars_pessimistic(plain, counts, n) == generate_cars(cenario, "Class", 0.05, 0.5, prune=True)


@pytest.mark.parametrize("cf", [0, 1, 1.5, float("nan")])
def test_invalid_prune_cf_fails_early(cenario, cf):
    with pytest.raises(ValueError):
        generate_cars(cenario, "Class", 0.05, 0.5, prune=True, prune_cf=cf)

