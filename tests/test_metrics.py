from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.typing import NDArray
from sklearn.metrics import f1_score, roc_auc_score

from metricci import f1, lift_at_k, roc_auc

Problem = tuple[NDArray[np.int64], NDArray[np.float64]]


def _binary_problem(draw: st.DrawFn, min_size: int = 4) -> Problem:
    n = draw(st.integers(min_size, 60))
    y = draw(arrays(np.int64, n, elements=st.integers(0, 1)))
    if y.min() == y.max():  # guarantee both classes
        y[0] = 1 - y[0]
    # Coarse scores on purpose, so ties are common.
    s = draw(arrays(np.float64, n, elements=st.integers(-5, 5).map(float)))
    return y, s


binary = st.composite(_binary_problem)


@given(binary())
def test_auc_matches_sklearn(problem: Problem) -> None:
    y, s = problem
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@given(binary())
def test_auc_is_bounded_and_antisymmetric(problem: Problem) -> None:
    y, s = problem
    a = float(roc_auc(y, s))
    assert 0.0 <= a <= 1.0
    assert float(roc_auc(y, -s)) == pytest.approx(1.0 - a, abs=1e-12)


@given(binary())
def test_auc_invariant_to_monotone_transform(problem: Problem) -> None:
    y, s = problem
    assert float(roc_auc(y, np.exp(s / 3))) == pytest.approx(float(roc_auc(y, s)), abs=1e-12)


@given(binary(), st.integers(0, 2**32 - 1))
def test_auc_invariant_to_row_order(problem: Problem, seed: int) -> None:
    y, s = problem
    perm = np.random.default_rng(seed).permutation(len(y))
    assert float(roc_auc(y[perm], s[perm])) == pytest.approx(float(roc_auc(y, s)), abs=1e-12)


def test_auc_without_both_classes_is_nan() -> None:
    assert math.isnan(float(roc_auc([1, 1, 1], [0.1, 0.2, 0.3])))


@given(binary())
def test_f1_matches_sklearn(problem: Problem) -> None:
    y, s = problem
    pred = (s > 0).astype(int)
    assert f1(y, pred) == pytest.approx(f1_score(y, pred, zero_division=0), abs=1e-12)


def test_f1_with_no_positives_anywhere_is_zero() -> None:
    assert f1([0, 0, 0], [0, 0, 0]) == 0.0


@given(binary(), st.floats(0.05, 1.0))
def test_lift_matches_explicit_loop(problem: Problem, k: float) -> None:
    y, s = problem
    top = max(1, math.ceil(k * len(y)))
    # Reference: walk rows in stable descending-score order.
    order = sorted(range(len(y)), key=lambda i: -s[i])
    hits = 0
    for i in order[:top]:
        hits += int(y[i])
    expected = (hits / top) / (y.sum() / len(y))
    assert float(lift_at_k(y, s, k)) == pytest.approx(expected, rel=1e-12)


def test_lift_of_perfect_ranking_is_inverse_base_rate() -> None:
    y = np.array([1] * 10 + [0] * 90)
    s = np.linspace(1, 0, 100)
    assert lift_at_k(y, s, 0.1) == pytest.approx(10.0)


def test_lift_rejects_bad_k() -> None:
    with pytest.raises(ValueError, match="k must"):
        lift_at_k([0, 1], [0.1, 0.2], 0.0)


@settings(max_examples=30)
@given(binary())
def test_stacked_rows_equal_single_calls(problem: Problem) -> None:
    y, s = problem
    rng = np.random.default_rng(0)
    idx = np.stack([rng.permutation(len(y)) for _ in range(4)])
    ys, ss = y[idx], s[idx]
    pred = (ss > 0).astype(int)
    for fn, second in ((roc_auc, ss), (f1, pred), (lift_at_k, ss)):
        stacked = np.asarray(fn(ys, second))
        single = [float(fn(ys[r], second[r])) for r in range(4)]
        np.testing.assert_allclose(stacked, single, rtol=1e-12)


def test_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        roc_auc([0, 2], [0.1, 0.2])
