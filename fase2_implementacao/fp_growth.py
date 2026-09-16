"""Algoritmo FP-Growth para mineração de itemsets frequentes.

Implementação própria do FP-Growth (Han, Pei & Yin, 2000), uma alternativa
ao Apriori (``apriori.py``) que evita a geração explícita de candidatos:
os dados são comprimidos em uma FP-Tree (árvore de padrões frequentes) e os
itemsets frequentes são extraídos recursivamente por meio de bases de
padrões condicionais, sem repetidas varreduras completas da base de dados a
cada tamanho de itemset.

Este módulo é independente do CBA-RG (que usa ``apriori.py`` como gerador
de itemsets, conforme a estrutura descrita no README do projeto). Ele existe
para permitir, em fases futuras do projeto, comparar o custo computacional
do Apriori com o do FP-Growth sobre o mesmo pipeline — ambos implementam a
mesma interface conceitual (itemset -> suporte), o que torna essa
comparação direta.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from fase2_implementacao.association_rules import Item, Itemset, Transaction


class _FPNode:
    """Nó de uma FP-Tree: um item, sua contagem, e a lista encadeada de nós
    do mesmo item (via ``node_link``), usada para localizar todas as
    ocorrências de um item na árvore ao montar sua base de padrões
    condicionais."""

    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item: Item | None, parent: "_FPNode | None"):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[Item, "_FPNode"] = {}
        self.node_link: "_FPNode | None" = None


def fp_growth_frequent_itemsets(
    transactions: Sequence[Transaction],
    min_support: float,
    max_length: int | None = None,
) -> dict[Itemset, float]:
    """Minera todos os itemsets frequentes de ``transactions`` via FP-Growth.

    A interface (itemset -> suporte) é a mesma de
    ``apriori.apriori_frequent_itemsets`` (exceto que este módulo não expõe
    ordem de geração, já que a mineração aqui é recursiva por sufixo, e não
    nível a nível como no Apriori) — os dois algoritmos devem produzir
    exatamente o mesmo conjunto de itemsets frequentes com os mesmos
    suportes para qualquer base de dados, o que é usado como verificação
    cruzada nos testes.
    """
    if not 0.0 <= min_support <= 1.0:
        raise ValueError("min_support deve estar entre 0 e 1")
    n = len(transactions)
    if n == 0:
        raise ValueError("não é possível minerar itemsets frequentes sem transações")

    frequent: dict[Itemset, float] = {}
    item_lists = [list(t) for t in transactions]
    _mine(item_lists, frozenset(), min_support, n, frequent, max_length)
    return frequent


def _count_items(transactions: list[list[Item]]) -> dict[Item, int]:
    counts: dict[Item, int] = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            counts[item] += 1
    return counts


def _build_tree(
    transactions: list[list[Item]], order: dict[Item, int]
) -> tuple[_FPNode, dict[Item, _FPNode]]:
    """Constrói a FP-Tree, inserindo em cada transação apenas os itens
    frequentes (presentes em ``order``), ordenados por frequência
    decrescente (itens mais frequentes ficam mais perto da raiz, o que
    maximiza o compartilhamento de prefixos entre transações)."""
    root = _FPNode(None, None)
    header: dict[Item, _FPNode] = {}
    tails: dict[Item, _FPNode] = {}
    for transaction in transactions:
        items = sorted((item for item in transaction if item in order), key=lambda item: order[item])
        node = root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = _FPNode(item, node)
                node.children[item] = child
                if item in header:
                    tails[item].node_link = child
                else:
                    header[item] = child
                tails[item] = child
            child.count += 1
            node = child
    return root, header


def _mine(
    transactions: list[list[Item]],
    prefix: frozenset[Item],
    min_support: float,
    n: int,
    frequent: dict[Itemset, float],
    max_length: int | None,
) -> None:
    """Passo recursivo: mina os itemsets frequentes que estendem ``prefix``
    dentro de ``transactions`` (a base de padrões condicionais de
    ``prefix``, ou as transações originais quando ``prefix`` é vazio)."""
    counts = _count_items(transactions)
    frequent_items = {item: count for item, count in counts.items() if count / n >= min_support}
    if not frequent_items:
        return

    # Ordem canônica por frequência decrescente; empates resolvidos pelo
    # próprio item, para tornar a mineração determinística.
    order = {
        item: rank
        for rank, (item, _) in enumerate(sorted(frequent_items.items(), key=lambda kv: (-kv[1], kv[0])))
    }
    root, header = _build_tree(transactions, order)

    # Processa do item menos frequente para o mais frequente (ordem de
    # sufixo padrão do FP-Growth): cada item vira a base de uma nova
    # recursão cujo prefixo é maior.
    items_by_suffix_order = sorted(frequent_items, key=lambda item: -order[item])
    for item in items_by_suffix_order:
        new_itemset = prefix | {item}
        frequent[new_itemset] = frequent_items[item] / n
        if max_length is not None and len(new_itemset) >= max_length:
            continue

        conditional_transactions: list[list[Item]] = []
        node = header[item]
        while node is not None:
            path: list[Item] = []
            ancestor = node.parent
            while ancestor is not None and ancestor.item is not None:
                path.append(ancestor.item)
                ancestor = ancestor.parent
            for _ in range(node.count):
                conditional_transactions.append(path)
            node = node.node_link

        if conditional_transactions:
            _mine(conditional_transactions, new_itemset, min_support, n, frequent, max_length)
