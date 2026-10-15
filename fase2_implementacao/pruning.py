"""Taxa de erro pessimista (estilo C4.5) usada na poda de regras do CBA-RG.

No artigo do CBA (Liu, Hsu e Ma, 1998), a função ``pruneRules`` usa o método de
poda por taxa de erro pessimista do C4.5 (Quinlan, 1992): se a taxa de erro
pessimista da regra ``r`` é maior que a da regra ``r-`` (obtida apagando uma
condição de ``r``), então ``r`` é podada. O artigo **não** dá a fórmula da taxa
pessimista nem o nível de confiança.

Escolha desta implementação (documentada, não vem do artigo)
------------------------------------------------------------
A taxa pessimista de uma regra que cobre ``n`` casos e erra ``e`` deles é o
**limite superior exato (Clopper--Pearson) da probabilidade de erro** ao nível
de confiança ``cf`` (padrão 0,25, o padrão do C4.5): o valor ``p`` tal que

    P(Binomial(n, p) <= e) = cf.

Casos particulares: ``e = n`` dá 1; ``e = 0`` dá ``1 - cf**(1/n)`` (forma
fechada). O C4.5 usa uma aproximação numérica desse mesmo limite (``AddErrs``);
aqui usa-se o valor exato por bisecção, sem dependências externas. A diferença
pode alterar poucos casos de fronteira e é uma escolha nossa, não do artigo.
"""

from __future__ import annotations

import math
from functools import lru_cache

DEFAULT_CF = 0.25


def _binomial_cdf(e: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= e), somando as probabilidades em escala logarítmica."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if e >= n else 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n_fact = math.lgamma(n + 1)
    total = 0.0
    for k in range(e + 1):
        total += math.exp(log_n_fact - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * log_p + (n - k) * log_q)
    return min(total, 1.0)


@lru_cache(maxsize=None)
def pessimistic_error_rate(errors: int, n: int, cf: float = DEFAULT_CF) -> float:
    """Taxa de erro pessimista: limite superior exato da probabilidade de erro (ver docstring do módulo)."""
    if isinstance(errors, bool) or isinstance(n, bool) or int(errors) != errors or int(n) != n:
        raise ValueError("errors e n devem ser inteiros")
    errors, n = int(errors), int(n)
    if n < 1:
        raise ValueError("n deve ser >= 1")
    if not 0 <= errors <= n:
        raise ValueError("errors deve estar em [0, n]")
    if isinstance(cf, bool) or not isinstance(cf, (int, float)) or not 0.0 < cf < 1.0:
        raise ValueError(f"cf deve estar em (0, 1), recebido: {cf!r}")
    if errors == n:
        return 1.0
    if errors == 0:
        return 1.0 - cf ** (1.0 / n)
    lo, hi = errors / n, 1.0  # a função P(Bin <= e) é decrescente em p; procura-se o p com valor cf
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _binomial_cdf(errors, n, mid) > cf:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2.0
