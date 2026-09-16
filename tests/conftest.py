"""Fixtures compartilhadas: datasets artificiais pequenos com valores que
podem ser conferidos manualmente, usados por todos os testes unitários da
Fase 1 (suporte/confiança/lift, Apriori, CBA-RG, CBA-CB M1)."""

from __future__ import annotations

import pytest

from fase2_implementacao.association_rules import Transaction


def _transaction(**items) -> Transaction:
    return frozenset(items.items())


@pytest.fixture
def cores_formas_transactions() -> list[Transaction]:
    """8 transações, 2 atributos categóricos (Color, Shape) + classe (Label).

    Desenhado para produzir suportes/confiança/lift em frações redondas,
    fáceis de conferir à mão:

    ==  =====  ======  =====
    id  Color  Shape   Label
    ==  =====  ======  =====
    1   Red    Circle  A
    2   Red    Circle  A
    3   Red    Square  A
    4   Red    Square  B
    5   Blue   Circle  B
    6   Blue   Circle  B
    7   Blue   Square  B
    8   Blue   Square  A
    ==  =====  ======  =====
    """
    rows = [
        ("Red", "Circle", "A"),
        ("Red", "Circle", "A"),
        ("Red", "Square", "A"),
        ("Red", "Square", "B"),
        ("Blue", "Circle", "B"),
        ("Blue", "Circle", "B"),
        ("Blue", "Square", "B"),
        ("Blue", "Square", "A"),
    ]
    return [_transaction(Color=color, Shape=shape, Label=label) for color, shape, label in rows]


@pytest.fixture
def cba_transactions() -> list[Transaction]:
    """10 transações, 1 atributo categórico (A) com 4 valores + classe (Class).

    Desenhado para que, com minsup=0.2 e minconf=0.5, o CBA-RG gere
    exatamente 3 CARs — duas delas empatadas em confiança e suporte (para
    testar o desempate por ordem de geração) — e para que o valor A=w não
    gere nenhuma CAR (por ficar abaixo do suporte mínimo em ambas as
    classes), deixando os dois exemplos com A=w descobertos por qualquer
    regra e sujeitos à classe padrão do CBA-CB M1:

    ===  =====
    A    Class
    ===  =====
    x    P
    x    P
    x    N
    y    N
    y    N
    y    P
    z    N
    z    N
    w    P
    w    N
    ===  =====
    """
    rows = [
        ("x", "P"),
        ("x", "P"),
        ("x", "N"),
        ("y", "N"),
        ("y", "N"),
        ("y", "P"),
        ("z", "N"),
        ("z", "N"),
        ("w", "P"),
        ("w", "N"),
    ]
    return [_transaction(A=a, Class=label) for a, label in rows]


@pytest.fixture
def cba_labels(cba_transactions) -> list[str]:
    return [dict(t)["Class"] for t in cba_transactions]
