"""Testes da discretização por entropia/MDL (Fayyad e Irani, 1993).

Os valores esperados são calculados à mão com ``math`` (independentes do código):

* ``x = 1..6``, classes ``aaabbb``: corte 3,5; Ent(S)=1; E=0; Gain=1;
  k=2, k1=k2=1: Delta = log2(7) - 2 = 0,8074; limiar = (log2(5) + 0,8074)/6 = 0,5215 < 1 -> aceito.
* ``x = 1..4``, classes ``abab``: melhor corte 1,5; Gain = 0,3113; limiar = 1,0572 -> rejeitado.
* ``x = 1..9``, classes ``aaabbbccc``: cortes 3,5 e 6,5 (ambos aceitos, em dois níveis).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fase2_implementacao.discretization_mdl import MDLDiscretizer, mdl_cut_points


def ent(counts):
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c)


def test_hand_computed_mdl_threshold_for_two_pure_halves():
    gain = 1.0
    delta = math.log2(3**2 - 2) - (2 * 1.0 - 0 - 0)
    threshold = (math.log2(6 - 1) + delta) / 6
    assert gain > threshold and threshold == pytest.approx(0.5215, abs=1e-3)  # confere o cálculo do docstring
    assert list(mdl_cut_points(np.arange(1, 7), np.array(list("aaabbb")))) == [3.5]


def test_alternating_classes_are_rejected_by_mdl():
    gain = ent([2, 2]) - (0.25 * 0 + 0.75 * ent([2, 1]))
    delta = math.log2(3**2 - 2) - (2 * ent([2, 2]) - 1 * 0 - 2 * ent([2, 1]))
    threshold = (math.log2(3) + delta) / 4
    assert gain < threshold  # logo o corte é rejeitado
    assert len(mdl_cut_points(np.arange(1, 5), np.array(list("abab")))) == 0


def test_three_classes_produce_two_cuts_in_two_recursion_levels():
    cuts = mdl_cut_points(np.arange(1, 10), np.array(list("aaabbbccc")))
    assert list(cuts) == [3.5, 6.5]


def test_cut_is_midpoint_between_distinct_values_and_ties_stay_together():
    x = np.array([1, 1, 1, 1, 4, 4, 4, 4], dtype=float)
    y = np.array(list("aaaabbbb"))
    assert list(mdl_cut_points(x, y)) == [2.5]


def test_values_with_same_value_but_different_classes_cannot_be_separated():
    x = np.array([5.0] * 10)
    y = np.array(list("aaaaabbbbb"))
    assert len(mdl_cut_points(x, y)) == 0  # coluna constante: uma faixa


def test_single_class_gives_no_cuts():
    assert len(mdl_cut_points(np.arange(10.0), np.array(["a"] * 10))) == 0


def test_cuts_do_not_depend_on_row_order():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = np.where(x + rng.normal(scale=0.3, size=60) > 0, "p", "n")
    perm = rng.permutation(60)
    assert np.array_equal(mdl_cut_points(x, y), mdl_cut_points(x[perm], y[perm]))


def test_separable_noise_free_data_gets_a_cut_between_the_classes():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(2, 3, 40)])
    y = np.array(["lo"] * 40 + ["hi"] * 40)
    cuts = mdl_cut_points(x, y)
    assert len(cuts) == 1 and 1 <= cuts[0] <= 2


def test_pure_noise_gets_no_cut():
    rng = np.random.default_rng(2)
    x = rng.normal(size=200)
    y = rng.choice(["a", "b"], size=200)
    assert len(mdl_cut_points(x, y)) == 0


# ------------------------------------------------------------- classe MDLDiscretizer ---


@pytest.fixture
def frame():
    return pd.DataFrame({"x": np.arange(1.0, 10.0), "color": list("rgbrgbrgb")}), np.array(list("aaabbbccc"))


def test_fit_transform_uses_labels_and_leaves_other_columns(frame):
    df, y = frame
    result = MDLDiscretizer(["x"]).fit_transform(df, y)
    assert list(result["x"]) == ["bin0"] * 3 + ["bin1"] * 3 + ["bin2"] * 3
    assert list(result["color"]) == list(df["color"])


def test_transform_needs_no_target_and_is_frozen_after_fit(frame):
    df, y = frame
    discretizer = MDLDiscretizer(["x"]).fit(df, y)
    out = discretizer.transform(pd.DataFrame({"x": [-100.0, 3.4, 3.5, 6.49, 6.5, 1000.0]}))
    assert list(out["x"]) == ["bin0", "bin0", "bin1", "bin1", "bin2", "bin2"]  # igual ao corte -> faixa superior


def test_target_of_test_data_cannot_change_the_transform(frame):
    df, y = frame
    discretizer = MDLDiscretizer(["x"]).fit(df, y)
    a = discretizer.transform(df)
    b = discretizer.transform(df.assign(label=list("cccbbbaaa")))  # coluna-alvo trocada
    assert list(a["x"]) == list(b["x"])


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        MDLDiscretizer(["x"]).transform(pd.DataFrame({"x": [1.0]}))


def test_categorical_only_dataset_works(frame):
    df, y = frame
    discretizer = MDLDiscretizer([]).fit(df[["color"]], y)
    pd.testing.assert_frame_equal(discretizer.transform(df[["color"]]), df[["color"]])


def test_nan_and_inf_are_rejected():
    y = np.array(list("aabb"))
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError, match="'x'"):
            MDLDiscretizer(["x"]).fit(pd.DataFrame({"x": [1.0, 2.0, bad, 4.0]}), y)
    discretizer = MDLDiscretizer(["x"]).fit(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}), y)
    with pytest.raises(ValueError):
        discretizer.transform(pd.DataFrame({"x": [np.nan]}))


def test_size_mismatch_and_empty_are_rejected():
    with pytest.raises(ValueError):
        MDLDiscretizer(["x"]).fit(pd.DataFrame({"x": [1.0, 2.0]}), ["a"])
    with pytest.raises(ValueError):
        MDLDiscretizer(["x"]).fit(pd.DataFrame({"x": []}), [])


def test_describe_reports_edges_and_occupancy(frame):
    df, y = frame
    d = MDLDiscretizer(["x"]).fit(df, y).describe(df)
    assert d["x"]["edges"] == [3.5, 6.5] and d["x"]["occupancy"] == {"bin0": 3, "bin1": 3, "bin2": 3}
    assert d["x"]["n_effective_bins"] == 3


# ----------------------------- implementação ingênua de referência (Python puro, sem numpy) ---


def _ref_cuts(pairs):
    """Fayyad e Irani direto das fórmulas, com laços simples; ``pairs`` = [(valor, classe)] ordenado."""
    pairs = sorted(pairs, key=lambda p: p[0])

    def counts(sub):
        out = {}
        for _, c in sub:
            out[c] = out.get(c, 0) + 1
        return list(out.values())

    def rec(sub):
        n = len(sub)
        if n < 2:
            return []
        e_s = ent(counts(sub))
        best = None
        for i in range(1, n):
            if sub[i - 1][0] == sub[i][0]:
                continue
            l, r = sub[:i], sub[i:]
            e = len(l) / n * ent(counts(l)) + len(r) / n * ent(counts(r))
            if best is None or e < best[0] - 1e-12:  # primeiro mínimo = menor corte
                best = (e, i)
        if best is None or e_s == 0:
            return []
        e, i = best
        l, r = sub[:i], sub[i:]
        k, k1, k2 = len(counts(sub)), len(counts(l)), len(counts(r))
        delta = math.log2(3**k - 2) - (k * e_s - k1 * ent(counts(l)) - k2 * ent(counts(r)))
        if (e_s - e) > (math.log2(n - 1) + delta) / n:
            return rec(l) + [(sub[i - 1][0] + sub[i][0]) / 2] + rec(r)
        return []

    return rec(pairs)


@pytest.mark.parametrize("seed", range(60))
def test_matches_naive_reference_on_random_data(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 120))
    n_classes = int(rng.integers(2, 5))
    signal = float(rng.choice([0.0, 0.5, 2.0]))
    y = rng.integers(0, n_classes, n)
    x = np.round(rng.normal(size=n) + signal * y, int(rng.choice([1, 2, 6])))  # arredondar cria valores repetidos
    labels = np.array([f"c{v}" for v in y])
    expected = _ref_cuts(list(zip(x.tolist(), labels.tolist())))
    assert mdl_cut_points(x, labels).tolist() == pytest.approx(expected)


def test_iris_full_dataset_cuts_match_the_classical_fayyad_irani_values():
    """Cortes MDL do Iris completo: 5,55/6,15; 2,95/3,35; 2,45/4,75; 0,8/1,75.

    São os valores clássicos desse dataset (saída conhecida do discretizador supervisionado MDL do
    Weka). Foram lembrados de memória, não confrontados com o Weka aqui; servem como teste de
    regressão: o código os reproduz exatamente.
    """
    from pathlib import Path

    iris = pd.read_csv(Path(__file__).resolve().parent.parent / "datasets" / "iris.csv")
    cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    edges = MDLDiscretizer(cols).fit(iris[cols], iris["Species"]).edges_
    assert {c: [round(float(e), 4) for e in edges[c]] for c in cols} == {
        "SepalLengthCm": [5.55, 6.15],
        "SepalWidthCm": [2.95, 3.35],
        "PetalLengthCm": [2.45, 4.75],
        "PetalWidthCm": [0.8, 1.75],
    }
