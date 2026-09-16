"""CBA: Classification Based on Associations (Liu, Hsu & Ma, 1998).

Implementação própria das duas etapas centrais do CBA:

- **CBA-RG** (Rule Generator, ``generate_cars``): gera as CARs (Class
  Association Rules) a partir dos itemsets frequentes minerados por
  ``apriori.apriori_frequent_itemsets``, restringindo o consequente a um
  único item de classe, e filtra por confiança mínima.
- **CBA-CB** (Classifier Builder, variante **M1**, ``build_classifier_m1``):
  ordena as CARs por precedência e constrói o classificador final por
  cobertura sequencial de banco de dados (database coverage).

O critério de ordenação/seleção (``sort_cars_by_precedence``) é passado como
parâmetro para ``build_classifier_m1`` justamente para manter CBA-RG e
CBA-CB desacoplados: uma futura extensão multicritério (CBA-QS, prevista
para a Fase 3 do projeto) poderá substituir apenas a função de ordenação,
sem alterar o gerador de CARs nem a lógica de cobertura.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fase2_implementacao.apriori import apriori_frequent_itemsets
from fase2_implementacao.association_rules import Item, Itemset, Transaction


@dataclass(frozen=True)
class Rule:
    """Uma Class Association Rule (CAR): antecedente ⇒ consequente (classe)."""

    antecedent: Itemset
    consequent: Item
    support: float
    confidence: float
    generation_order: int

    @property
    def class_label(self) -> Any:
        return self.consequent[1]

    def matches(self, transaction: Transaction) -> bool:
        """True se a transação satisfaz o antecedente da regra."""
        return self.antecedent <= transaction

    def __len__(self) -> int:
        """Comprimento do antecedente (número de condições atributo=valor)."""
        return len(self.antecedent)


def generate_cars(
    transactions: Sequence[Transaction],
    class_attribute: str,
    min_support: float,
    min_confidence: float,
    max_antecedent_length: int | None = None,
) -> list[Rule]:
    """CBA-RG: gera as CARs válidas a partir das transações de treino.

    Minera todos os itemsets frequentes (via Apriori) e mantém apenas
    aqueles que contêm exatamente um item de classe — esse item vira o
    consequente da regra, e o restante do itemset vira o antecedente.
    Itemsets com zero ou mais de um item de classe não correspondem a CARs
    válidas e são descartados; itemsets cujo único conteúdo é o item de
    classe (antecedente vazio) também são descartados, pois uma CAR precisa
    de ao menos uma condição.

    O suporte do antecedente sozinho é obtido do próprio dicionário de
    itemsets frequentes: por downward closure, se ``antecedente ∪ {classe}``
    é frequente, o antecedente também é (é um subconjunto seu), portanto já
    foi calculado e armazenado pelo Apriori.

    Parameters
    ----------
    max_antecedent_length:
        Limite opcional no número de condições do antecedente (equivale a
        um limite de ``max_antecedent_length + 1`` no tamanho do itemset
        minerado, já que o item de classe ocupa uma posição).
    """
    max_itemset_length = None if max_antecedent_length is None else max_antecedent_length + 1
    frequent, order = apriori_frequent_itemsets(transactions, min_support, max_itemset_length)

    cars: list[Rule] = []
    for itemset, itemset_support in frequent.items():
        class_items = [item for item in itemset if item[0] == class_attribute]
        if len(class_items) != 1:
            continue
        consequent = class_items[0]
        antecedent = itemset - {consequent}
        if not antecedent:
            continue
        antecedent_support = frequent[antecedent]
        if antecedent_support == 0.0:
            continue
        conf = itemset_support / antecedent_support
        if conf < min_confidence:
            continue
        cars.append(
            Rule(
                antecedent=antecedent,
                consequent=consequent,
                support=itemset_support,
                confidence=conf,
                generation_order=order[itemset],
            )
        )
    return cars


def sort_cars_by_precedence(cars: Sequence[Rule]) -> list[Rule]:
    """Ordena CARs pelo critério de precedência clássico do CBA (Liu, Hsu & Ma, 1998).

    Uma regra r1 tem precedência sobre r2 se:
    1. conf(r1) > conf(r2); ou
    2. conf(r1) == conf(r2) e sup(r1) > sup(r2); ou
    3. conf(r1) == conf(r2), sup(r1) == sup(r2), e r1 foi gerada antes de r2
       (ou seja, tem antecedente mais curto, ou mesmo tamanho mas descoberta
       antes na varredura do Apriori).

    Esta é a única função que uma extensão multicritério (CBA-QS) precisaria
    substituir para mudar o critério de seleção sem tocar em ``generate_cars``
    nem em ``build_classifier_m1``.
    """
    return sorted(cars, key=lambda r: (-r.confidence, -r.support, r.generation_order))


def majority_class(labels: Sequence[Any]) -> Any:
    """Classe mais frequente em ``labels``; empates são resolvidos por ordem alfabética.

    A ordem alfabética como desempate é uma escolha arbitrária, mas
    determinística — necessária para que o classificador seja reprodutível
    mesmo nos casos (raros) em que duas classes empatam em frequência entre
    os exemplos não cobertos por nenhuma regra.
    """
    if not labels:
        raise ValueError("não é possível determinar a classe majoritária de uma lista vazia")
    counts: dict[Any, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    max_count = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == max_count)
    return tied[0]


@dataclass
class Classifier:
    """Classificador CBA final: lista de decisão ordenada + classe padrão."""

    rules: list[Rule]
    default_class: Any
    class_attribute: str

    def predict(self, transaction: Transaction) -> Any:
        """Aplica a lista de decisão: primeira regra compatível decide a classe."""
        for rule in self.rules:
            if rule.matches(transaction):
                return rule.class_label
        return self.default_class

    def predict_many(self, transactions: Sequence[Transaction]) -> list[Any]:
        return [self.predict(t) for t in transactions]


def build_classifier_m1(
    cars: Sequence[Rule],
    transactions: Sequence[Transaction],
    labels: Sequence[Any],
    class_attribute: str,
    sort_fn: Callable[[Sequence[Rule]], list[Rule]] = sort_cars_by_precedence,
) -> Classifier:
    """CBA-CB, variante M1: cobertura de banco de dados sequencial.

    Para cada regra, na ordem de precedência dada por ``sort_fn``: entre os
    exemplos de treino ainda não cobertos por nenhuma regra anterior,
    verifica quais satisfazem o antecedente da regra. Se pelo menos um
    desses exemplos tiver o rótulo correto (isto é, a regra classifica
    corretamente pelo menos um caso ainda não coberto), a regra é
    adicionada ao classificador e **todos** os exemplos que satisfazem o
    antecedente (corretos ou não) são marcados como cobertos e removidos
    das rodadas seguintes. Regras que não classificam corretamente nenhum
    exemplo não coberto são descartadas (não entram no classificador, e não
    removem nenhum exemplo).

    Ao final, a classe padrão é a classe majoritária entre os exemplos que
    permaneceram não cobertos por qualquer regra selecionada; se todos os
    exemplos foram cobertos, usa-se a classe majoritária geral do conjunto
    de treino.

    Esta é a variante M1 de Liu, Hsu & Ma (1998): cobertura sequencial
    simples, sem a etapa adicional de truncar o classificador no ponto de
    erro mínimo acumulado que o artigo original também descreve — a
    especificação desta reprodução (Fase 1 do projeto) usa apenas a
    cobertura sequencial com classe padrão ao final, deixando variantes
    mais elaboradas (M2, poda por erro mínimo) para fases posteriores.
    """
    if len(transactions) != len(labels):
        raise ValueError("transactions e labels devem ter o mesmo tamanho")

    ordered_cars = sort_fn(cars)
    covered: set[int] = set()
    selected: list[Rule] = []

    for rule in ordered_cars:
        uncovered_matching = [
            i for i in range(len(transactions)) if i not in covered and rule.matches(transactions[i])
        ]
        if not uncovered_matching:
            continue
        correctly_classified = [i for i in uncovered_matching if labels[i] == rule.class_label]
        if not correctly_classified:
            continue
        selected.append(rule)
        covered.update(uncovered_matching)

    remaining_labels = [labels[i] for i in range(len(labels)) if i not in covered]
    default = majority_class(remaining_labels if remaining_labels else labels)

    return Classifier(rules=selected, default_class=default, class_attribute=class_attribute)
