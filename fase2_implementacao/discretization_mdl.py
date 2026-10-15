"""Discretização supervisionada por entropia com critério MDL (Fayyad e Irani, 1993).

É o método de discretização usado nos experimentos do artigo original do CBA
(Liu, Hsu e Ma, 1998: ``Discretization of continuous attributes is done using
the Entropy method in (Fayyad and Irani 1993)``). Este módulo é uma
*alternativa opcional* ao ``EqualFrequencyDiscretizer``; a baseline por quantis
não foi alterada.

Algoritmo (por atributo, recursivo)
-----------------------------------
Dado um conjunto ``S`` de ``N`` exemplos ordenados pelo valor do atributo:

1. Candidatos a corte: pontos médios entre valores *distintos* consecutivos.
   (Fayyad e Irani provam que o corte ótimo está num ponto de fronteira entre
   classes; avaliar todos os pontos médios entre valores distintos encontra o
   mesmo mínimo, só com mais candidatos.)
2. Escolhe-se o corte ``T`` que minimiza a entropia de classe ponderada

   ``E(T; S) = |S1|/N * Ent(S1) + |S2|/N * Ent(S2)``,

   em que ``S1`` são os exemplos com valor ``< T`` e ``S2`` os demais. Empates
   (diferença < 1e-12) resolvem-se pelo menor corte, para ser determinístico.
3. O corte só é aceito se passar no critério MDL (descrição de comprimento
   mínimo):

   ``Gain(T; S) > ( log2(N - 1) + Delta(T; S) ) / N``

   ``Gain = Ent(S) - E(T; S)`` e
   ``Delta = log2(3**k - 2) - [ k*Ent(S) - k1*Ent(S1) - k2*Ent(S2) ]``,
   onde ``k``, ``k1`` e ``k2`` são os números de classes presentes em ``S``,
   ``S1`` e ``S2``. Entropias em base 2.
4. Se aceito, repete-se em ``S1`` e ``S2``; senão, o intervalo não é dividido.

O número de faixas é determinado pelos dados (não é um parâmetro). Um
atributo sem corte aceito vira uma única faixa (``bin0``): não discrimina as
classes pelo critério MDL. Um atributo constante também tem uma só faixa.

Convenção de fronteiras e vazamento
-----------------------------------
Igual ao discretizador por quantis: ``bin_i = [e_{i-1}, e_i)``; valor igual a
um corte vai para a faixa superior; valores fora do intervalo de treino caem
na faixa extrema mais próxima. Os cortes são aprendidos com ``X`` **e ``y`` do
treino** de cada dobra (``fit``); a transformação (``transform``) é única por
atributo e **não usa o alvo**, podendo ser aplicada a qualquer ``X`` novo --- é
o tipo de discretização supervisionada legítima descrita no achado A04 do
Relatório 02 (cortes globais aprendidos no treino e congelados). Como no
discretizador por quantis, ``NaN`` e infinitos são rejeitados.
"""

from __future__ import annotations

import numpy as np


def _entropy(counts: np.ndarray) -> np.ndarray:
    """Entropia (base 2) das distribuições de classe em ``counts`` (última dimensão = classes)."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, counts / total, 0.0)
        terms = np.where(counts > 0, p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -terms.sum(axis=-1)


def mdl_cut_points(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cortes MDL (Fayyad e Irani) de um atributo; devolve um vetor ordenado (possivelmente vazio)."""
    values = np.asarray(values, dtype=float)
    _, codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    n_classes = int(codes.max()) + 1 if len(codes) else 0
    order = np.argsort(values, kind="stable")
    v, c = values[order], codes[order]
    n = len(v)
    cum = np.zeros((n + 1, max(n_classes, 1)))
    if n:
        cum[1:] = np.cumsum(np.eye(n_classes)[c], axis=0)

    cuts: list[float] = []
    stack = [(0, n)]
    while stack:
        lo, hi = stack.pop()
        m = hi - lo
        if m < 2:
            continue
        total = cum[hi] - cum[lo]
        ent = float(_entropy(total))
        if ent == 0.0:
            continue  # intervalo puro: nada a separar
        split = np.arange(lo + 1, hi)
        split = split[v[split - 1] != v[split]]  # só entre valores distintos
        if len(split) == 0:
            continue
        left, right = cum[split] - cum[lo], cum[hi] - cum[split]
        e_split = ((split - lo) * _entropy(left) + (hi - split) * _entropy(right)) / m
        best = int(np.flatnonzero(e_split <= e_split.min() + 1e-12)[0])  # menor corte entre empates
        i = int(split[best])
        gain = ent - float(e_split[best])
        k = int((total > 0).sum())
        k1, k2 = int((left[best] > 0).sum()), int((right[best] > 0).sum())
        delta = np.log2(3**k - 2) - (k * ent - k1 * float(_entropy(left[best])) - k2 * float(_entropy(right[best])))
        threshold = (np.log2(m - 1) + delta) / m
        if gain > threshold:
            cut = (v[i - 1] + v[i]) / 2.0
            if not v[i - 1] < cut:  # arredondamento: garante v[i-1] < corte <= v[i]
                cut = float(v[i])
            cuts.append(float(cut))
            stack.append((lo, i))
            stack.append((i, hi))
    return np.array(sorted(cuts), dtype=float)


class MDLDiscretizer:
    """Discretizador supervisionado por entropia/MDL, ajustado por coluna.

    Parameters
    ----------
    columns:
        Colunas contínuas a discretizar; as demais passam inalteradas. Pode ser vazia.
    """

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.edges_: dict[str, np.ndarray] = {}
        self.n_effective_bins_: dict[str, int] = {}
        self.is_fitted_ = False

    @staticmethod
    def _finite_values(df, column: str) -> np.ndarray:
        if column not in df.columns:
            raise ValueError(f"coluna {column!r} não encontrada no DataFrame")
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"a coluna {column!r} contém valores ausentes (NaN) ou infinitos; "
                "a discretização exige valores finitos (trate-os antes)"
            )
        return values

    def fit(self, df, y) -> "MDLDiscretizer":
        """Aprende os cortes de cada coluna com ``df`` e os rótulos ``y`` do TREINO."""
        if len(df) == 0:
            raise ValueError("não é possível ajustar o discretizador com zero linhas")
        y = np.asarray(list(y))
        if len(y) != len(df):
            raise ValueError("y deve ter o mesmo número de linhas que df")
        edges = {c: mdl_cut_points(self._finite_values(df, c), y) for c in self.columns}
        self.edges_ = edges
        self.n_effective_bins_ = {c: len(e) + 1 for c, e in edges.items()}
        self.is_fitted_ = True
        return self

    def transform(self, df):
        """Aplica os cortes já aprendidos (não usa o alvo). Devolve uma cópia com faixas ``"bin0"``, ``"bin1"``, ..."""
        if not self.is_fitted_:
            raise RuntimeError("MDLDiscretizer.fit precisa ser chamado antes de transform")
        out = df.copy()
        for column in self.columns:
            idx = np.searchsorted(self.edges_[column], self._finite_values(df, column), side="right")
            out[column] = [f"bin{i}" for i in idx]
        return out

    def fit_transform(self, df, y):
        return self.fit(df, y).transform(df)

    def describe(self, df=None) -> dict:
        """Cortes, convenção de fronteira e (se ``df`` for dado) ocupação dos bins, para auditoria."""
        report = {}
        for column in self.columns:
            entry = {
                "method": "entropy-mdl (Fayyad e Irani, 1993)",
                "edges": [float(e) for e in self.edges_[column]],
                "n_effective_bins": self.n_effective_bins_[column],
                "boundary": "bin_i = [edge_{i-1}, edge_i); valor igual ao corte vai para a faixa superior",
                "missing_policy": "rejeitar (ValueError)",
            }
            if df is not None:
                idx = np.searchsorted(self.edges_[column], self._finite_values(df, column), side="right")
                entry["occupancy"] = {f"bin{i}": int((idx == i).sum()) for i in range(self.n_effective_bins_[column])}
            report[column] = entry
        return report
