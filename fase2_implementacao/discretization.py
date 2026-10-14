"""Discretização de atributos contínuos por igual-frequência (quantis).

O CBA opera sobre pares atributo=valor categóricos; atributos numéricos
contínuos (no Iris: ``SepalLengthCm``, ``SepalWidthCm``, ``PetalLengthCm``,
``PetalWidthCm``) precisam ser transformados em faixas discretas antes da
codificação transacional.

Esta é a discretização *baseline* da Fase 1 do projeto: igual-frequência
(equal-frequency / quantis), com número de faixas (bins) configurável — por
exemplo 3 (tercis) ou 4 (quartis). Cada faixa recebe aproximadamente a mesma
quantidade de exemplos de treino. A comparação com discretização
supervisionada (que usa os rótulos de classe para escolher os pontos de
corte) fica para a Fase 2 do plano de progressão, conforme
``docs/PIBIC___João_e_Fábio.pdf``.

Convenção de fronteiras
-----------------------
Com os cortes internos ``e_0 < e_1 < ... < e_{m-1}``, a faixa ``bin_i`` é
``[e_{i-1}, e_i)``: a primeira é ``x < e_0``, a última é ``x >= e_{m-1}`` e
**um valor exatamente igual a um corte vai para a faixa superior**
(``numpy.searchsorted(..., side="right")``). Os limites externos são,
conceitualmente, -infinito e +infinito: um valor de teste fora do intervalo
observado no treino cai na faixa extrema mais próxima.

Cortes degenerados (achado A07 do Relatório 02)
-----------------------------------------------
Quantis de colunas com muitos valores repetidos podem coincidir com o mínimo
ou cair no mesmo vão entre dois valores distintos consecutivos, o que criaria
faixas vazias. Por isso cada corte de quantil é reposicionado para um vão
entre valores *distintos* observados no treino, e cortes que caem no mesmo
vão são colapsados em um só. Consequências:

- Valores iguais ficam sempre na mesma faixa.
- Uma coluna constante tem zero cortes e um único bin (``bin0``).
- O número efetivo de faixas pode ser menor que ``n_bins`` (igual-frequência
  é aproximada na presença de empates); ele fica em ``n_effective_bins_``.
- Toda faixa efetiva tem pelo menos um exemplo no treino.

Valores ausentes (achado A03)
-----------------------------
A política da baseline é **rejeitar**: ``NaN`` e ``±inf`` (em ``fit`` ou
``transform``) levantam ``ValueError`` com o nome da coluna. Assim nenhum
corte contém NaN e uma ausência nunca é confundida com um valor alto.
Imputação, se houver, deve ser ajustada só no treino, fora desta classe.

Vazamento
---------
Os cortes são calculados apenas com os dados de treino (o ``fit`` deve ser
chamado uma vez por dobra de validação cruzada, nunca com o dataset
completo). A transformação é única por atributo e **independe do alvo**: o
mesmo conjunto de cortes é aplicado a qualquer ``X`` novo, sem acesso à classe.

NÃO adotar a proposta de discretização separada por classe do último
parágrafo de ``docs/relatorio_pokemon.tex`` (achado A04): escolher os cortes
pela classe *verdadeira* de cada caso exigiria conhecer o alvo antes de
prever (vazamento direto) e o mesmo rótulo ``bin2`` passaria a significar
intervalos diferentes em classes diferentes. Uma discretização supervisionada
legítima aprende cortes globais com ``X`` e ``y`` do treino e os congela.
"""

from __future__ import annotations

import numpy as np


class EqualFrequencyDiscretizer:
    """Discretizador por igual-frequência (quantis), ajustado por coluna.

    Parameters
    ----------
    n_bins:
        Número de faixas desejado por coluna (inteiro >= 2; ex.: 3 para
        tercis, 4 para quartis). Se a coluna tiver poucos valores distintos
        ou muitos empates, o número efetivo de faixas é menor (ver docstring
        do módulo) — esperado e documentado, não tratado como erro.
    columns:
        Lista de nomes de colunas contínuas a discretizar. As demais
        colunas do DataFrame passam por ``transform`` inalteradas. Pode ser
        vazia (base só-categórica): ``transform`` devolve então uma cópia
        idêntica.
    """

    def __init__(self, n_bins: int, columns: list[str]):
        if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)):
            raise ValueError(f"n_bins deve ser inteiro, recebido: {n_bins!r}")
        if n_bins < 2:
            raise ValueError("n_bins deve ser >= 2")
        self.n_bins = int(n_bins)
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

    def _cut_points(self, values: np.ndarray) -> np.ndarray:
        distinct = np.unique(values)
        if len(distinct) < 2:
            return np.array([], dtype=float)  # constante: um único bin
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        raw_edges = np.quantile(values, quantiles)
        edges: dict[int, float] = {}
        for edge in raw_edges:
            # gap = nº de valores distintos estritamente menores que o corte:
            # o corte separa distinct[gap-1] (abaixo) de distinct[gap] (>= corte).
            gap = int(np.searchsorted(distinct, edge, side="left"))
            if gap == 0:  # corte <= mínimo geraria faixa vazia: sobe para o 1º vão
                gap, edge = 1, distinct[1]
            edges.setdefault(gap, float(edge))  # cortes no mesmo vão colapsam
        return np.array([edges[g] for g in sorted(edges)], dtype=float)

    def fit(self, df) -> "EqualFrequencyDiscretizer":
        """Calcula os pontos de corte internos de cada coluna a partir de ``df``.

        Deve ser chamado apenas com dados de treino.
        """
        if len(df) == 0:
            raise ValueError("não é possível ajustar o discretizador com zero linhas")
        edges_by_column: dict[str, np.ndarray] = {}
        for column in self.columns:
            edges_by_column[column] = self._cut_points(self._finite_values(df, column))
        self.edges_ = edges_by_column
        self.n_effective_bins_ = {column: len(edges) + 1 for column, edges in edges_by_column.items()}
        self.is_fitted_ = True
        return self

    def transform(self, df):
        """Aplica as faixas já ajustadas a ``df`` (treino ou teste).

        Retorna uma cópia do DataFrame com as colunas discretizadas
        substituídas por rótulos de faixa (strings ``"bin0"``, ``"bin1"``,
        ...). Levanta ``RuntimeError`` se chamado antes de ``fit`` e
        ``ValueError`` se houver NaN/infinito nas colunas contínuas.
        """
        if not self.is_fitted_:
            raise RuntimeError("EqualFrequencyDiscretizer.fit precisa ser chamado antes de transform")
        out = df.copy()
        for column in self.columns:
            values = self._finite_values(df, column)
            bin_indices = np.searchsorted(self.edges_[column], values, side="right")
            out[column] = [f"bin{i}" for i in bin_indices]
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def describe(self, df=None) -> dict:
        """Cortes, convenção de fronteira e (se ``df`` for dado) ocupação dos bins, para auditoria."""
        report = {}
        for column in self.columns:
            entry = {
                "edges": [float(e) for e in self.edges_[column]],
                "n_effective_bins": self.n_effective_bins_[column],
                "boundary": "bin_i = [edge_{i-1}, edge_i); valor igual ao corte vai para a faixa superior",
                "missing_policy": "rejeitar (ValueError)",
            }
            if df is not None:
                indices = np.searchsorted(self.edges_[column], self._finite_values(df, column), side="right")
                entry["occupancy"] = {
                    f"bin{i}": int((indices == i).sum()) for i in range(self.n_effective_bins_[column])
                }
            report[column] = entry
        return report
