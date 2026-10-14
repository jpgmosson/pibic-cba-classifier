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


# ------------------------------------------------------------------ Relatório 02: A03, A07, A08 ---

import numpy as np  # noqa: E402


def test_nan_and_inf_in_fit_are_rejected_with_column_name():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="'x'"):
            EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(pd.DataFrame({"x": [1.0, bad, 3.0]}))


def test_nan_in_transform_is_rejected_not_sent_to_a_value_bin(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(train_df)
    with pytest.raises(ValueError, match="ausentes"):
        discretizer.transform(pd.DataFrame({"x": [1.0, np.nan]}))


def test_fitted_edges_never_contain_nan(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=4, columns=["x"]).fit(train_df)
    assert np.isfinite(discretizer.edges_["x"]).all()


def test_degenerate_quantiles_preserve_distinction_between_two_values():
    df = pd.DataFrame({"x": [0] * 8 + [1] * 2})
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(df)
    assert discretizer.n_effective_bins_["x"] == 2
    result = discretizer.transform(df)
    assert list(result["x"]) == ["bin0"] * 8 + ["bin1"] * 2


def test_constant_column_has_single_bin_and_new_values_do_not_open_empty_bins():
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(pd.DataFrame({"x": [5, 5, 5]}))
    assert len(discretizer.edges_["x"]) == 0 and discretizer.n_effective_bins_["x"] == 1
    assert list(discretizer.transform(pd.DataFrame({"x": [1, 5, 9]}))["x"]) == ["bin0", "bin0", "bin0"]


def test_every_effective_bin_is_occupied_in_training_even_with_heavy_ties():
    df = pd.DataFrame({"x": [1] + [10] * 9, "y": [0] * 5 + [1, 2, 3, 4, 5]})
    discretizer = EqualFrequencyDiscretizer(n_bins=4, columns=["x", "y"]).fit(df)
    for column, entry in discretizer.describe(df).items():
        assert all(count > 0 for count in entry["occupancy"].values()), (column, entry)
        assert len(entry["occupancy"]) == entry["n_effective_bins"]


def test_boundary_convention_value_equal_to_cut_goes_to_upper_bin(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(train_df)
    cut0, cut1 = discretizer.edges_["x"]
    eps = 1e-9
    result = discretizer.transform(pd.DataFrame({"x": [cut0 - eps, cut0, cut0 + eps, cut1 - eps, cut1]}))
    assert list(result["x"]) == ["bin0", "bin1", "bin1", "bin1", "bin2"]


def test_categorical_only_dataset_with_no_columns_works_after_fit():
    df = pd.DataFrame({"color": ["red", "blue"], "shape": ["a", "b"]})
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=[])
    with pytest.raises(RuntimeError):  # antes do fit continua inválido
        discretizer.transform(df)
    result = discretizer.fit(df).transform(df)
    pd.testing.assert_frame_equal(result, df)
    assert result is not df


def test_is_fitted_is_independent_of_edges(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=[])
    assert not discretizer.is_fitted_
    assert discretizer.fit(train_df).is_fitted_ and discretizer.edges_ == {}


def test_n_bins_must_be_a_real_integer():
    for bad in (True, 2.5, "3"):
        with pytest.raises(ValueError):
            EqualFrequencyDiscretizer(n_bins=bad, columns=["x"])


def test_fit_with_zero_rows_and_missing_column_are_rejected():
    with pytest.raises(ValueError):
        EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(pd.DataFrame({"x": []}))
    with pytest.raises(ValueError, match="não encontrada"):
        EqualFrequencyDiscretizer(n_bins=3, columns=["z"]).fit(pd.DataFrame({"x": [1, 2, 3]}))


def test_transform_does_not_need_the_target_column(train_df):
    discretizer = EqualFrequencyDiscretizer(n_bins=3, columns=["x"]).fit(train_df)
    assert list(discretizer.transform(pd.DataFrame({"x": [1, 5, 9]})).columns) == ["x"]
