"""Algoritmo Apriori clássico para mineração de itemsets frequentes.

Implementação própria do Apriori (Agrawal & Srikant, 1994), na forma
adaptada por Liu, Hsu & Ma (1998) para o CBA-RG: geração de candidatos
nível a nível, com poda por *downward closure* (todo subconjunto de um
itemset frequente também é frequente; logo, um candidato com algum
subconjunto infrequente pode ser descartado sem contar seu suporte).

Este módulo é agnóstico ao domínio do CBA — não sabe o que é um atributo de
classe, apenas minera itemsets frequentes genéricos sobre itens
``(atributo, valor)``. A restrição a CARs (regras de associação de classe)
é feita em ``cba.py``, que usa ``apriori_frequent_itemsets`` como base para
o CBA-RG.

Ordem de geração
----------------
Além do suporte, cada itemset frequente recebe um índice de *ordem de
geração*: a posição em que foi confirmado como frequente, seguindo a ordem
natural do Apriori (todos os itemsets de tamanho 1 primeiro, em ordem
canônica; depois os de tamanho 2, na ordem em que os candidatos foram
testados; e assim por diante). Essa ordem é usada pelo CBA-RG como critério
de desempate final na precedência das regras, conforme Liu, Hsu & Ma (1998).
"""

from __future__ import annotations

from collections.abc import Sequence

from fase2_implementacao.association_rules import Item, Itemset, Transaction, support


def apriori_frequent_itemsets(
    transactions: Sequence[Transaction],
    min_support: float,
    max_length: int | None = None,
) -> tuple[dict[Itemset, float], dict[Itemset, int]]:
    """Minera todos os itemsets frequentes de ``transactions``.

    Parameters
    ----------
    transactions:
        Sequência de transações (frozensets de itens).
    min_support:
        Suporte mínimo (fração entre 0 e 1) para um itemset ser considerado
        frequente.
    max_length:
        Tamanho máximo de itemset a minerar. ``None`` significa sem limite
        (a mineração para naturalmente quando nenhum candidato novo passa
        no suporte mínimo).

    Returns
    -------
    frequent:
        Dicionário itemset -> suporte, com **todos** os itemsets frequentes
        de todos os tamanhos (não apenas os maximais) — isso é necessário
        porque o CBA-RG precisa do suporte do antecedente sozinho (um
        subconjunto de tamanho k-1 de um itemset de tamanho k) para
        calcular confiança, e por downward closure ele está sempre presente
        neste dicionário.
    order:
        Dicionário itemset -> índice de ordem de geração (inteiro
        crescente, começando em 0), na ordem em que cada itemset foi
        confirmado como frequente.
    """
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support deve estar entre 0 e 1")
    if not transactions:
        raise ValueError("não é possível minerar itemsets frequentes sem transações")

    frequent: dict[Itemset, float] = {}
    order: dict[Itemset, int] = {}
    next_order = 0

    # Nível 1: conta cada item individualmente.
    item_counts: dict[Item, int] = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1

    n = len(transactions)
    current_level: list[Itemset] = []
    for item in sorted(item_counts):
        supp = item_counts[item] / n
        if supp >= min_support:
            itemset = frozenset((item,))
            frequent[itemset] = supp
            order[itemset] = next_order
            next_order += 1
            current_level.append(itemset)
    current_level.sort(key=lambda s: sorted(s))

    size = 1
    while current_level and (max_length is None or size < max_length):
        candidates = _generate_candidates(current_level, frequent)
        next_level: list[Itemset] = []
        for candidate in candidates:
            supp = support(candidate, transactions)
            if supp >= min_support:
                frequent[candidate] = supp
                order[candidate] = next_order
                next_order += 1
                next_level.append(candidate)
        current_level = next_level
        size += 1

    return frequent, order


def _generate_candidates(
    previous_level: list[Itemset], frequent: dict[Itemset, float]
) -> list[Itemset]:
    """Gera candidatos de tamanho k a partir dos itemsets frequentes de tamanho k-1.

    Passo de junção clássico do Apriori: dois itemsets de tamanho k-1 são
    unidos se, ordenados canonicamente, compartilham os primeiros k-2 itens
    (para k=2, todo par é unido). O candidato resultante é descartado
    (poda por downward closure) se algum de seus subconjuntos de tamanho
    k-1 não estiver em ``frequent``.
    """
    sorted_itemsets = [tuple(sorted(itemset)) for itemset in previous_level]
    sorted_itemsets.sort()
    k_minus_1 = len(sorted_itemsets[0]) if sorted_itemsets else 0

    candidates: list[Itemset] = []
    seen: set[Itemset] = set()
    n = len(sorted_itemsets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sorted_itemsets[i], sorted_itemsets[j]
            if a[: k_minus_1 - 1] != b[: k_minus_1 - 1]:
                # Como a lista está ordenada, uma vez que o prefixo comum
                # deixa de bater não há mais parceiros possíveis para `a`.
                break
            candidate = frozenset(a) | frozenset(b)
            if len(candidate) != k_minus_1 + 1 or candidate in seen:
                continue
            seen.add(candidate)
            if _all_subsets_frequent(candidate, frequent):
                candidates.append(candidate)
    return candidates


def _all_subsets_frequent(candidate: Itemset, frequent: dict[Itemset, float]) -> bool:
    """Poda por downward closure: todo subconjunto de tamanho k-1 deve ser frequente."""
    for item in candidate:
        subset = candidate - {item}
        if subset not in frequent:
            return False
    return True
