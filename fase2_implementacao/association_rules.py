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

import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any

Item = tuple[str, Any]
Itemset = frozenset[Item]
Transaction = frozenset[Item]


def to_fraction(value: float, name: str = "valor") -> Fraction:
    """Converte um limiar decimal de configuração (ex.: ``0.05``) em ``Fraction`` exata.

    A semântica adotada é a *decimal*: ``0.05`` significa 5/100, e não o
    ``float`` binário mais próximo. Isso permite comparar contagens inteiras
    com o limiar sem erro de ponto flutuante (ex.: ``3/10 >= 0.3`` é sempre
    verdadeiro). Rejeita booleanos, NaN e infinitos.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(f"{name} deve ser numérico, recebido: {value!r}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} deve ser finito, recebido: {value!r}")
    return Fraction(str(value)) if isinstance(value, float) else Fraction(value)


def min_count_for(min_support: float, n: int) -> int:
    """Menor contagem inteira ``c`` com ``c / n >= min_support`` (limiar decimal exato)."""
    threshold = to_fraction(min_support, "min_support")
    return -((-threshold.numerator * n) // threshold.denominator)


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
    return _rows_to_transactions(df, columns)


def dataframe_to_feature_transactions(df, class_column: str | None = None) -> list[Transaction]:
    """Codifica apenas os preditores de ``df`` (sem a coluna de classe).

    Usada na **inferência**: a classe verdadeira nunca entra na transação
    que o classificador enxerga (achados A04/A11 do Relatório 02), de modo
    que remover ou permutar os rótulos de teste não altera as predições.
    Se ``class_column`` for informada e existir em ``df``, ela é ignorada.
    """
    columns = [c for c in df.columns if c != class_column]
    return _rows_to_transactions(df, columns)


def _rows_to_transactions(df, columns: list[str]) -> list[Transaction]:
    # Ausência nunca vira a categoria textual "nan" (achado A09): é rejeitada.
    for col in columns:
        if df[col].isna().any():
            raise ValueError(f"a coluna {col!r} contém valores ausentes (NaN/None); trate-os antes da codificação")
    # Conversão coluna a coluna (e não linha a linha com iterrows): evita que o
    # pandas promova inteiros a float ao montar a linha de uma base com tipos
    # mistos, o que faria o mesmo valor virar "45" no treino e "45.0" no teste.
    values_by_column = {col: df[col].astype(str).tolist() for col in columns}
    return [frozenset((col, values_by_column[col][i]) for col in columns) for i in range(len(df))]
