"""Testes da discretização por igual-frequência.

Com os valores 1..9 (9 pontos) e n_bins=3, os cortes de quantil (método
linear do numpy) ficam em aproximadamente 3.667 e 6.333, produzindo tercis
exatos: {1,2,3} -> bin0, {4,5,6} -> bin1, {7,8,9} -> bin2. Também testamos
que valores fora da faixa de treino (bem menores ou bem maiores) caem no
bin extremo mais próximo, conforme a escolha registrada em
``discretization.py`` (limites externos ilimitados)."""

from __future__ import annotations

import pandas as pd
import pytest

from fase2_implementacao.discretization import EqualFrequencyDiscretizer


@pytest.fixture
def train_df():
    return pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9]})


def test_fit_transform_produces_equal_frequency_tercis(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"])
    result = discretizer.fit_transform(train_df)

    assert list(result["x"]) == ["bin0"] * 3 + ["bin1"] * 3 + ["bin2"] * 3


def test_transform_before_fit_raises():
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"])
    with pytest.raises(RuntimeError):
        discretizer.transform(pd.DataFrame({"x": [1, 2, 3]}))


def test_out_of_range_values_fall_into_nearest_outer_bin(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(train_df)

    test_df = pd.DataFrame({"x": [-1000, 0.5, 3.7, 1000]})
    result = discretizer.transform(test_df)

    assert list(result["x"]) == ["bin0", "bin0", "bin1", "bin2"]


def test_n_bins_below_2_is_rejected():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretizer(n_bins=1, columns=["x"])
