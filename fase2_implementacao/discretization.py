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

Tratamento de valores fora da faixa de treino
----------------------------------------------
Os pontos de corte são calculados apenas com os dados de treino (o
``fit`` deve ser chamado uma vez por dobra de validação cruzada, nunca com o
dataset completo, para evitar vazamento de informação do teste — ver seção
7.3 do relatório do projeto). Um valor de teste fora do intervalo observado
no treino é atribuído à faixa extrema mais próxima: os limites externos são,
conceitualmente, -infinito e +infinito. Isso é obtido apenas considerando os
pontos de corte *internos* (as fronteiras entre faixas) e usando busca
binária (``numpy.searchsorted``), sem nunca comparar contra um limite
mínimo/máximo explícito — qualquer valor, por mais extremo, cai na primeira
ou na última faixa.
"""

from __future__ import annotations

import numpy as np


class EqualFrequencyDiscretizer:
    """Discretizador por igual-frequência (quantis), ajustado por coluna.

    Parameters
    ----------
    n_bins:
        Número de faixas desejado por coluna (ex.: 3 para tercis, 4 para
        quartis). Se a coluna tiver poucos valores distintos, pontos de
        corte de quantis coincidentes são colapsados, resultando em menos
        de ``n_bins`` faixas efetivas — isso é esperado e documentado aqui
        em vez de tratado como erro.
    columns:
        Lista de nomes de colunas contínuas a discretizar. As demais
        colunas do DataFrame passam por ``transform`` inalteradas.
    """

    def __init__(self, n_bins: int, columns: list[str]):
        if n_bins < 2:
            raise ValueError("n_bins deve ser >= 2")
        self.n_bins = n_bins
        self.columns = list(columns)
        self.edges_: dict[str, np.ndarray] = {}

    def fit(self, df) -> "EqualFrequencyDiscretizer":
        """Calcula os pontos de corte internos de cada coluna a partir de ``df``.

        Deve ser chamado apenas com dados de treino.
        """
        for column in self.columns:
            values = df[column].to_numpy(dtype=float)
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(values, quantiles))
            self.edges_[column] = edges
        return self

    def transform(self, df):
        """Aplica as faixas já ajustadas a ``df`` (treino ou teste).

        Retorna uma cópia do DataFrame com as colunas discretizadas
        substituídas por rótulos de faixa (strings ``"bin0"``, ``"bin1"``,
        ...). Levanta ``RuntimeError`` se chamado antes de ``fit``.
        """
        if not self.edges_:
            raise RuntimeError("EqualFrequencyDiscretizer.fit precisa ser chamado antes de transform")
        out = df.copy()
        for column in self.columns:
            edges = self.edges_[column]
            values = df[column].to_numpy(dtype=float)
            # searchsorted com os pontos de corte internos: valores <= edges[0]
            # caem no bin 0, valores > edges[-1] caem no último bin — os
            # limites externos são, na prática, -inf/+inf.
            bin_indices = np.searchsorted(edges, values, side="right")
            out[column] = [f"bin{i}" for i in bin_indices]
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)
