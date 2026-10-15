"""CBA: Classification Based on Associations (Liu, Hsu & Ma, 1998).

Implementação própria das duas etapas centrais do CBA:

- **CBA-RG** (Rule Generator, ``generate_cars``): gera as CARs (Class
  Association Rules) a partir dos itemsets frequentes minerados por
  ``apriori.apriori_frequent_itemset_counts``. Como no artigo, para cada
  antecedente (*condset*) é mantido **um único** consequente — o ruleitem de
  maior confiança (a regra possível, PR) — e a CAR só existe se essa
  confiança atingir ``min_confidence``.
- **CBA-CB** (Classifier Builder, variante **M1**, ``build_classifier_m1``):
  ordena as CARs por precedência, seleciona-as por cobertura sequencial de
  banco de dados e, como na Figura 2 do artigo, **trunca a lista no
  primeiro prefixo de menor erro total** (erros das regras + erros da
  classe padrão residual), usando a classe padrão registrada nesse prefixo.

Todas as contagens (conjunta, do antecedente, N) são inteiras e as razões
(confiança, suporte) são comparadas de forma exata com ``fractions.Fraction``
— nunca por ``float`` — para que empates matemáticos não sejam quebrados por
arredondamento (achado A10 do Relatório 02).

O critério de ordenação (``sort_cars_by_precedence``) é passado como
parâmetro para ``build_classifier_m1`` para manter CBA-RG e CBA-CB
desacoplados.

Desvios deliberados em relação ao artigo (documentados):

- Empate de confiança entre consequentes do mesmo antecedente: o artigo
  sorteia; aqui o desempate é determinístico (maior contagem conjunta e, se
  ainda empatar, ordem alfabética do rótulo da classe).
- Se todos os casos de treino forem cobertos pelas regras selecionadas, a
  classe padrão (sem casos residuais para estimá-la) é a classe majoritária
  de todo o treino (empate: ordem alfabética).
- O prefixo vazio (nenhuma regra) não concorre ao corte, como na Figura 2:
  só prefixos com pelo menos uma regra são avaliados. Se nenhuma regra for
  selecionada, o classificador é só a classe majoritária do treino.
- Não há poda pessimista de condições (opção distinta, não implementada).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from fase2_implementacao.apriori import apriori_frequent_itemset_counts
from fase2_implementacao.association_rules import Item, Itemset, Transaction, to_fraction
from fase2_implementacao.pruning import DEFAULT_CF, pessimistic_error_rate


@dataclass(frozen=True)
class Rule:
    """Uma Class Association Rule (CAR): antecedente ⇒ consequente (classe).

    Guarda as contagens inteiras de origem; ``support`` e ``confidence``
    (``float``) são derivados delas e ``*_fraction`` dão a razão exata.
    """

    antecedent: Itemset
    consequent: Item
    joint_count: int
    antecedent_count: int
    n_transactions: int
    generation_order: int

    @property
    def support(self) -> float:
        return self.joint_count / self.n_transactions

    @property
    def confidence(self) -> float:
        return self.joint_count / self.antecedent_count

    @property
    def support_fraction(self) -> Fraction:
        return Fraction(self.joint_count, self.n_transactions)

    @property
    def confidence_fraction(self) -> Fraction:
        return Fraction(self.joint_count, self.antecedent_count)

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
    prune: bool = False,
    prune_cf: float = DEFAULT_CF,
) -> list[Rule]:
    """CBA-RG: gera as CARs válidas a partir das transações de treino.

    Minera os itemsets frequentes (Apriori) e considera como *ruleitems* os
    que contêm exatamente um item de classe (o consequente) e ao menos uma
    condição (o antecedente). Como em Liu, Hsu & Ma (1998), entre todos os
    ruleitems com o mesmo antecedente é escolhida **uma só** regra possível —
    a de maior confiança (mesmo antecedente ⇒ mesma contagem de
    antecedente, logo maior confiança = maior contagem conjunta; empate:
    ordem alfabética do rótulo, escolha determinística em vez do sorteio do
    artigo). A regra só vira CAR se sua confiança for >= ``min_confidence``.

    Limiares são avaliados exatamente (decimal, ver ``to_fraction``). A
    lista retornada está em ordem de geração do Apriori.

    Parameters
    ----------
    max_antecedent_length:
        Limite opcional no número de condições do antecedente (equivale a
        um limite de ``max_antecedent_length + 1`` no tamanho do itemset
        minerado, já que o item de classe ocupa uma posição).
    prune:
        Se verdadeiro, aplica a poda pessimista de condições do artigo
        (``pruneRules``, ver ``prune_cars_pessimistic``) às CARs geradas. Padrão
        falso: sem poda (as ``CARs`` do artigo, não as ``prCARs``).
    prune_cf:
        Nível de confiança da taxa de erro pessimista (padrão 0,25, do C4.5).
    """
    min_sup = to_fraction(min_support, "min_support")
    min_conf = to_fraction(min_confidence, "min_confidence")
    if not 0 < min_sup <= 1:
        raise ValueError(f"min_support deve estar em (0, 1], recebido: {min_support!r}")
    if not 0 <= min_conf <= 1:
        raise ValueError(f"min_confidence deve estar em [0, 1], recebido: {min_confidence!r}")
    if prune:
        pessimistic_error_rate(0, 1, prune_cf)  # valida prune_cf (0 < cf < 1) antes de minerar
    if max_antecedent_length is not None and (
        isinstance(max_antecedent_length, bool)
        or not isinstance(max_antecedent_length, int)
        or max_antecedent_length < 1
    ):
        raise ValueError(f"max_antecedent_length deve ser None ou inteiro >= 1, recebido: {max_antecedent_length!r}")

    max_itemset_length = None if max_antecedent_length is None else max_antecedent_length + 1
    counts, order, n = apriori_frequent_itemset_counts(transactions, min_support, max_itemset_length)

    # Um vencedor por antecedente: maior contagem conjunta; empate -> menor rótulo.
    best: dict[Itemset, tuple[Item, int]] = {}
    for itemset, joint in counts.items():
        class_items = [item for item in itemset if item[0] == class_attribute]
        if len(class_items) != 1:
            continue
        consequent = class_items[0]
        antecedent = itemset - {consequent}
        if not antecedent:
            continue
        current = best.get(antecedent)
        if current is None or joint > current[1] or (joint == current[1] and str(consequent[1]) < str(current[0][1])):
            best[antecedent] = (consequent, joint)

    cars: list[Rule] = []
    for antecedent, (consequent, joint) in best.items():
        antecedent_count = counts[antecedent]
        if Fraction(joint, antecedent_count) < min_conf:
            continue
        cars.append(
            Rule(
                antecedent=antecedent,
                consequent=consequent,
                joint_count=joint,
                antecedent_count=antecedent_count,
                n_transactions=n,
                generation_order=order[antecedent | {consequent}],
            )
        )
    cars.sort(key=lambda r: r.generation_order)
    if prune:
        cars = prune_cars_pessimistic(cars, counts, n, prune_cf)
    return cars


def prune_cars_pessimistic(
    cars: Sequence[Rule],
    counts: dict[Itemset, int],
    n: int,
    cf: float = DEFAULT_CF,
) -> list[Rule]:
    """Poda pessimista de condições (``pruneRules`` do CBA-RG, Liu, Hsu e Ma, 1998).

    Uma regra ``r`` é podada se sua taxa de erro pessimista for **estritamente
    maior** que a de alguma regra ``r-`` obtida apagando UMA condição de ``r``
    (mesmo consequente). A taxa vem de ``pessimistic_error_rate(erros, casos)``,
    com ``casos`` = nº de transações que satisfazem o antecedente e ``erros`` =
    casos dentre eles de outra classe. Decisões de interpretação (o artigo não
    as especifica):

    * ``r-`` é qualquer regra com uma condição a menos; se ``r`` tem uma única
      condição, ``r-`` é a regra de antecedente vazio (prevê a classe com a
      prevalência do treino, ``casos`` = N).
    * Basta UMA ``r-`` melhor para podar ``r``; empates não podam.
    * A comparação usa as contagens de todos os itemsets frequentes, mesmo de
      regras que não passaram em ``min_confidence`` (``r-`` pode ter confiança
      baixa); por downward closure, as contagens necessárias existem.

    A ordem relativa das CARs sobreviventes é preservada.
    """
    kept: list[Rule] = []
    for rule in cars:
        errors = rule.antecedent_count - rule.joint_count
        rate = pessimistic_error_rate(errors, rule.antecedent_count, cf)
        pruned = False
        for condition in rule.antecedent:
            shorter = rule.antecedent - {condition}
            if shorter:
                shorter_n = counts[shorter]
                shorter_joint = counts[shorter | {rule.consequent}]
            else:
                shorter_n = n
                shorter_joint = counts[frozenset({rule.consequent})]
            if rate > pessimistic_error_rate(shorter_n - shorter_joint, shorter_n, cf):
                pruned = True
                break
        if not pruned:
            kept.append(rule)
    return kept


def sort_cars_by_precedence(cars: Sequence[Rule]) -> list[Rule]:
    """Ordena CARs pelo critério de precedência clássico do CBA (Liu, Hsu & Ma, 1998).

    Uma regra r1 tem precedência sobre r2 se:
    1. conf(r1) > conf(r2); ou
    2. conf(r1) == conf(r2) e sup(r1) > sup(r2); ou
    3. conf(r1) == conf(r2), sup(r1) == sup(r2), e r1 foi gerada antes de r2
       (ou seja, tem antecedente mais curto, ou mesmo tamanho mas descoberta
       antes na varredura do Apriori).

    Confiança e suporte são comparados como frações exatas (contagens
    inteiras), não como ``float``: 2/6 e 3/9 empatam em confiança e a
    segunda vence pelo maior suporte, sem interferência de arredondamento.
    """
    return sorted(cars, key=lambda r: (-r.confidence_fraction, -r.support_fraction, r.generation_order))


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


@dataclass(frozen=True)
class PrefixRecord:
    """Estado do M1 após selecionar a ``n_rules``-ésima regra (uma linha do histórico de erro)."""

    n_rules: int
    rule: Rule
    rule_errors: int  # erros acumulados das regras do prefixo (cada caso contado só na regra que o cobre)
    n_remaining: int  # casos de treino não cobertos pelo prefixo
    default_class: Any  # maioria dos casos residuais
    default_errors: int  # erros da classe padrão nos casos residuais
    total_errors: int  # rule_errors + default_errors


@dataclass
class Classifier:
    """Classificador CBA final: lista de decisão ordenada + classe padrão.

    ``m1_history`` guarda um :class:`PrefixRecord` por regra selecionada
    *antes* do corte (inclusive as descartadas depois dele) e
    ``chosen_prefix_length`` o tamanho do prefixo escolhido, de modo que o
    corte do M1 seja auditável.
    """

    rules: list[Rule]
    default_class: Any
    class_attribute: str
    m1_history: list[PrefixRecord] = field(default_factory=list)
    chosen_prefix_length: int = 0

    def predict_with_rule(self, transaction: Transaction) -> tuple[Any, int | None]:
        """Classe prevista e índice da regra acionada (``None`` = classe padrão)."""
        for index, rule in enumerate(self.rules):
            if rule.matches(transaction):
                return rule.class_label, index
        return self.default_class, None

    def predict(self, transaction: Transaction) -> Any:
        """Aplica a lista de decisão: primeira regra compatível decide a classe."""
        return self.predict_with_rule(transaction)[0]

    def predict_many(self, transactions: Sequence[Transaction]) -> list[Any]:
        return [self.predict(t) for t in transactions]


def build_classifier_m1(
    cars: Sequence[Rule],
    transactions: Sequence[Transaction],
    labels: Sequence[Any],
    class_attribute: str,
    sort_fn: Callable[[Sequence[Rule]], list[Rule]] = sort_cars_by_precedence,
) -> Classifier:
    """CBA-CB, variante M1 (Figura 2 de Liu, Hsu & Ma, 1998).

    Para cada regra, na ordem de precedência dada por ``sort_fn``: entre os
    casos de treino ainda não cobertos, verifica quais satisfazem o
    antecedente. Se a regra classifica corretamente ao menos um deles, ela
    entra na lista C e **todos** os casos que satisfazem o antecedente
    (corretos ou não) são removidos. Após cada inserção registra-se:

    - ``rule_errors``: erros acumulados das regras de C (cada caso é contado
      uma única vez, na regra que o cobre);
    - a classe padrão = maioria dos casos residuais e seus erros
      (``default_errors``);
    - ``total_errors = rule_errors + default_errors``.

    Ao final, mantém-se o **primeiro** prefixo de C com o menor
    ``total_errors`` (empates conservam o prefixo mais curto), descartam-se
    as regras seguintes e usa-se a classe padrão registrada nesse prefixo.
    Sem nenhuma regra selecionada, o classificador é só a classe majoritária
    do treino. Ver o docstring do módulo para os demais desvios documentados.
    """
    if len(transactions) != len(labels):
        raise ValueError("transactions e labels devem ter o mesmo tamanho")
    if len(labels) == 0:
        raise ValueError("o conjunto de treino não pode ser vazio")

    ordered_cars = sort_fn(cars)
    overall_default = majority_class(labels)
    uncovered = list(range(len(transactions)))
    selected: list[Rule] = []
    history: list[PrefixRecord] = []
    rule_errors = 0

    for rule in ordered_cars:
        matching = [i for i in uncovered if rule.matches(transactions[i])]
        if not matching:
            continue
        wrong = sum(1 for i in matching if labels[i] != rule.class_label)
        if wrong == len(matching):  # não acerta nenhum caso residual
            continue
        selected.append(rule)
        rule_errors += wrong
        matched = set(matching)
        uncovered = [i for i in uncovered if i not in matched]

        if uncovered:
            residual_labels = [labels[i] for i in uncovered]
            default_class = majority_class(residual_labels)
            default_errors = sum(1 for label in residual_labels if label != default_class)
        else:
            default_class, default_errors = overall_default, 0
        history.append(
            PrefixRecord(
                n_rules=len(selected),
                rule=rule,
                rule_errors=rule_errors,
                n_remaining=len(uncovered),
                default_class=default_class,
                default_errors=default_errors,
                total_errors=rule_errors + default_errors,
            )
        )

    if not history:
        return Classifier(rules=[], default_class=overall_default, class_attribute=class_attribute)

    best = history[0]
    for record in history[1:]:
        if record.total_errors < best.total_errors:  # estrito: o primeiro mínimo vence
            best = record
    return Classifier(
        rules=selected[: best.n_rules],
        default_class=best.default_class,
        class_attribute=class_attribute,
        m1_history=history,
        chosen_prefix_length=best.n_rules,
    )
