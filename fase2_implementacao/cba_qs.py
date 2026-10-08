"""CBA-QS: seleção/ordenação de regras por qualidade e simplicidade (Fase 3).

Implementa a extensão proposta em ``docs/PIBIC___João_e_Fábio.pdf``, seção
5.3: um escore multicritério

    S(r) = w1 * Q̃(r) + w2 * Ĩ(r) + w3 * C̃(r) - w4 * L̃(r)

que substitui apenas o critério de ordenação/seleção de regras do CBA-CB.
O CBA-RG (geração de CARs, em ``apriori.py``/``cba.generate_cars``)
permanece idêntico, e a cobertura sequencial de banco de dados
(``cba.build_classifier_m1``) também é reaproveitada — este módulo só troca
a função de ordenação (``sort_fn``) que ela recebe.

O documento define cada termo apenas por exemplo ("por exemplo, confiança
corrigida ou fator de certeza" para Q; "por exemplo, lift ou risco
relativo" para I), sem fixar uma fórmula única, um método de normalização
ou pesos padrão. As escolhas abaixo foram decididas explicitamente com o
autor do projeto (não inferidas) para tornar a fórmula implementável:

- **Q(r)** — fator de certeza (certainty factor), que corrige a confiança
  pela prevalência da classe consequente (ver ``certainty_factor``).
- **I(r)** — lift(r) = conf(r) / sup(consequente), a mesma definição de
  ``association_rules.lift``.
- **C(r)** — cobertura = sup(antecedente), a fração das transações de
  treino que satisfazem X independentemente da classe (o documento diz
  "cobertura", não confiança — a confiança já entra em Q).
- **L(r)** — comprimento do antecedente, ``len(rule)``.
- **Normalização** — min-max, calculada entre as CARs candidatas do
  próprio fold de treino (nunca usando dados de teste), antes de combinar
  os quatro termos.
- **Pesos w1..w4** — o documento não sugere valores; o padrão adotado é
  peso igual (0.25 cada), documentado aqui como decisão explícita a ser
  revisada nas fases seguintes (ver ``DEFAULT_WEIGHTS``). São
  configuráveis via YAML no runner de experimentos.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fase2_implementacao.association_rules import Transaction, support
from fase2_implementacao.cba import Classifier, Rule, build_classifier_m1

DEFAULT_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def certainty_factor(confidence: float, prior: float) -> float:
    """Fator de certeza: confiança corrigida pela prevalência da classe.

    CF(r) = (conf - prior) / (1 - prior)  se conf >= prior
    CF(r) = (conf - prior) / prior         caso contrário

    Limitado a [-1, 1]; CF = 0 quando a regra não é melhor que prever a
    classe pela sua prevalência (conf == prior). Retorna 0.0 nos casos
    degenerados (prior == 0 ou prior == 1) em que o denominador seria zero.
    """
    if confidence >= prior:
        denom = 1.0 - prior
        return (confidence - prior) / denom if denom > 0 else 0.0
    return (confidence - prior) / prior if prior > 0 else 0.0


def _min_max_normalize(values: dict[Rule, float]) -> dict[Rule, float]:
    """Normaliza um dicionário regra->valor bruto para [0, 1] (min-max).

    Quando todas as CARs têm o mesmo valor bruto (faixa zero), esse
    critério não discrimina nenhuma regra das outras, então todas recebem
    0.5 (ponto médio neutro) em vez de uma divisão por zero.
    """
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return {rule: 0.5 for rule in values}
    return {rule: (value - lo) / (hi - lo) for rule, value in values.items()}


@dataclass(frozen=True)
class ScoredRule:
    """Uma CAR com seu escore S(r) do CBA-QS já calculado."""

    rule: Rule
    score: float


def score_cars(
    cars: Sequence[Rule],
    transactions: Sequence[Transaction],
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> list[ScoredRule]:
    """Calcula S(r) para cada CAR em ``cars``.

    ``transactions`` deve ser o mesmo conjunto de treino usado pelo CBA-RG
    para gerar as CARs (nunca o teste) — tanto os priors de classe quanto a
    normalização min-max usam apenas esses dados, para não vazar informação
    do teste na etapa de seleção de regras.
    """
    if not cars:
        return []
    w1, w2, w3, w4 = weights

    priors: dict[Any, float] = {}
    for rule in cars:
        if rule.consequent not in priors:
            priors[rule.consequent] = support({rule.consequent}, transactions)

    raw_quality = {rule: certainty_factor(rule.confidence, priors[rule.consequent]) for rule in cars}
    raw_interest = {
        rule: (rule.confidence / priors[rule.consequent] if priors[rule.consequent] > 0 else 0.0) for rule in cars
    }
    raw_coverage = {rule: support(rule.antecedent, transactions) for rule in cars}
    raw_length = {rule: float(len(rule)) for rule in cars}

    quality = _min_max_normalize(raw_quality)
    interest = _min_max_normalize(raw_interest)
    coverage = _min_max_normalize(raw_coverage)
    length = _min_max_normalize(raw_length)

    return [
        ScoredRule(
            rule=rule,
            score=w1 * quality[rule] + w2 * interest[rule] + w3 * coverage[rule] - w4 * length[rule],
        )
        for rule in cars
    ]


def sort_cars_by_score(
    cars: Sequence[Rule],
    transactions: Sequence[Transaction],
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> list[Rule]:
    """Ordena as CARs por S(r) decrescente — critério de precedência do CBA-QS.

    Empates em S(r) são desempatados pela ordem de geração do Apriori,
    mesmo critério final de desempate usado por
    ``cba.sort_cars_by_precedence``, para manter o resultado determinístico.
    """
    scored = score_cars(cars, transactions, weights)
    ordered = sorted(scored, key=lambda sr: (-sr.score, sr.rule.generation_order))
    return [sr.rule for sr in ordered]


def build_classifier_cba_qs(
    cars: Sequence[Rule],
    transactions: Sequence[Transaction],
    labels: Sequence[Any],
    class_attribute: str,
    weights: tuple[float, float, float, float] = DEFAULT_WEIGHTS,
) -> Classifier:
    """CBA-CB usando o critério de seleção do CBA-QS.

    Reaproveita ``cba.build_classifier_m1`` (mesma cobertura sequencial de
    banco de dados e mesma lógica de classe padrão), trocando apenas
    ``sort_fn`` pelo escore S(r) do CBA-QS — nenhuma lógica de cobertura é
    duplicada ou reimplementada aqui.
    """
    sort_fn = functools.partial(sort_cars_by_score, transactions=transactions, weights=weights)
    return build_classifier_m1(cars, transactions, labels, class_attribute, sort_fn=sort_fn)
