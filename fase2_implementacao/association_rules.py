"""Cálculo de métricas de regras de associação.

Este módulo é responsável pelas métricas fundamentais usadas na mineração de
regras de associação e na construção do classificador CBA: suporte (support),
confiança (confidence) e lift, conforme definidas em Liu, Hsu & Ma (1998),
"Integrating Classification and Association Rule Mining", KDD-98.

Representação de dados
-----------------------
Uma *transação* é um ``frozenset`` de *itens*, e um item é um par
``(atributo, valor)`` (uma tupla de dois elementos). O rótulo de classe é
tratado como um item especial cujo atributo é o nome da coluna de classe —
não há nada estruturalmente diferente entre um item de classe e um item de
atributo comum; a distinção só importa para quem monta as CARs (ver
``cba.py``). Um *itemset* (antecedente, consequente, ou qualquer subconjunto)
também é representado como ``frozenset`` de itens, o que permite usar o
operador de subconjunto (``<=``) do Python diretamente para testar se uma
transação satisfaz um itemset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

Item = tuple[str, Any]
Itemset = frozenset[Item]
Transaction = frozenset[Item]


def _as_itemset(itemset: Iterable[Item]) -> Itemset:
    """Normaliza qualquer iterável de itens para um ``frozenset``."""
    return itemset if isinstance(itemset, frozenset) else frozenset(itemset)


def support(itemset: Iterable[Item], transactions: Sequence[Transaction]) -> float:
    """Suporte de um itemset: fração das transações que o contêm.

    sup(X) = |{t em D : X ⊆ t}| / |D|

    Por convenção, o suporte do itemset vazio é 1.0 (todo registro "contém"
    o conjunto vazio de condições).
    """
    if not transactions:
        raise ValueError("não é possível calcular suporte sem transações")
    itemset = _as_itemset(itemset)
    if not itemset:
        return 1.0
    count = sum(1 for t in transactions if itemset <= t)
    return count / len(transactions)


def confidence(
    antecedent: Iterable[Item],
    consequent: Iterable[Item],
    transactions: Sequence[Transaction],
) -> float:
    """Confiança da regra antecedente ⇒ consequente.

    conf(X ⇒ Y) = sup(X ∪ Y) / sup(X)

    Estima a proporção de registros que satisfazem o antecedente X e também
    satisfazem o consequente Y. Se sup(X) = 0 (nenhuma transação contém o
    antecedente), a confiança é matematicamente indefinida; por convenção
    retornamos 0.0 nesse caso, já que não há evidência alguma que sustente a
    regra.
    """
    antecedent = _as_itemset(antecedent)
    consequent = _as_itemset(consequent)
    antecedent_support = support(antecedent, transactions)
    if antecedent_support == 0.0:
        return 0.0
    combined_support = support(antecedent | consequent, transactions)
    return combined_support / antecedent_support


def lift(
    antecedent: Iterable[Item],
    consequent: Iterable[Item],
    transactions: Sequence[Transaction],
) -> float:
    """Lift da regra antecedente ⇒ consequente.

    lift(X ⇒ Y) = conf(X ⇒ Y) / sup(Y)

    Razão entre a confiança observada e a confiança esperada sob
    independência entre X e Y. lift < 1 indica correlação negativa,
    lift = 1 indica independência, lift > 1 indica correlação positiva. Se
    sup(Y) = 0, o lift é indefinido; por convenção retornamos 0.0.
    """
    consequent = _as_itemset(consequent)
    consequent_support = support(consequent, transactions)
    if consequent_support == 0.0:
        return 0.0
    return confidence(antecedent, consequent, transactions) / consequent_support


def dataframe_to_transactions(df, class_column: str) -> list[Transaction]:
    """Codifica um DataFrame em uma lista de transações (codificação transacional).

    Cada linha vira uma transação: um ``frozenset`` com um item
    ``(nome_da_coluna, valor)`` por coluna, incluindo a coluna de classe.
    Os valores são convertidos para ``str`` para garantir que sejam
    hasheáveis e comparáveis de forma estável (importante para a ordenação
    canônica usada pelo Apriori em ``apriori.py``), independentemente do
    dtype original da coluna (numérico, categórico já discretizado, etc.).

    Atributos contínuos devem ser discretizados (ver ``discretization.py``)
    antes de chamar esta função — este módulo não faz nenhuma suposição
    sobre o tipo dos valores, apenas os trata como categorias.
    """
    columns = list(df.columns)
    if class_column not in columns:
        raise ValueError(f"coluna de classe {class_column!r} não encontrada no DataFrame")
    transactions: list[Transaction] = []
    for _, row in df.iterrows():
        transactions.append(frozenset((col, str(row[col])) for col in columns))
    return transactions
