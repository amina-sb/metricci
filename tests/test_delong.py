from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from metricci import bootstrap_ci, roc_auc
from metricci.delong import _components, delong_ci, delong_test


def _pairwise_reference(
    y: NDArray[np.int64], s: NDArray[np.float64]
) -> tuple[list[float], list[float]]:
    """The O(n1 * n0) definition from DeLong et al. (1988), written as loops."""
    pos = [float(v) for v, t in zip(s, y, strict=True) if t == 1]
    neg = [float(v) for v, t in zip(s, y, strict=True) if t == 0]

    def psi(x: float, z: float) -> float:
        return 1.0 if x > z else 0.5 if x == z else 0.0

    v10 = []
    for x in pos:
        total = 0.0
        for z in neg:
            total += psi(x, z)
        v10.append(total / len(neg))
    v01 = []
    for z in neg:
        total = 0.0
        for x in pos:
            total += psi(x, z)
        v01.append(total / len(pos))
    return v10, v01


def _data(
    seed: int, n_pos: int = 25, n_neg: int = 45
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    # Rounded scores so ties occur.
    raw = np.r_[rng.normal(1, 1, n_pos), rng.normal(0, 1, n_neg)]
    s = np.asarray(np.round(raw, 1), dtype=np.float64)
    return y, s


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_components_match_pairwise_definition(seed: int) -> None:
    y, s = _data(seed)
    auc, v10, v01 = _components(y, s)
    ref10, ref01 = _pairwise_reference(y, s)
    np.testing.assert_allclose(v10, ref10, atol=1e-12)
    np.testing.assert_allclose(v01, ref01, atol=1e-12)
    assert auc == pytest.approx(float(roc_auc(y, s)), abs=1e-12)


def test_variance_is_close_to_the_stratified_bootstrap() -> None:
    # Both estimate the same sampling variance; at n = 400 they should agree to
    # within Monte Carlo error of the bootstrap (a few percent of the width).
    y, s = _data(3, 120, 280)
    d = delong_ci(y, s)

    def auc_v(t: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(roc_auc(t, u), dtype=np.float64)

    b = bootstrap_ci(auc_v, y, s, method="percentile", n_resamples=8000, strata=y,
                     vectorized=True, random_state=0)  # fmt: skip
    assert d.high - d.low == pytest.approx(b.high - b.low, rel=0.08)


def test_identical_models_give_pvalue_one() -> None:
    y, s = _data(4)
    res = delong_test(y, s, s.copy())
    assert res.difference == 0.0
    assert res.pvalue == 1.0


def test_paired_test_detects_a_clear_difference() -> None:
    rng = np.random.default_rng(5)
    y = np.r_[np.ones(100, dtype=np.int64), np.zeros(200, dtype=np.int64)]
    shared = rng.normal(0, 1, 300)
    a = y * 1.5 + shared + rng.normal(0, 0.5, 300)
    b = y * 0.5 + shared + rng.normal(0, 0.5, 300)
    res = delong_test(y, a, b)
    assert res.difference > 0
    assert res.pvalue < 1e-6


def test_test_is_antisymmetric() -> None:
    y, s = _data(6)
    other = s + np.random.default_rng(6).normal(0, 0.7, s.shape[0])
    ab, ba = delong_test(y, s, other), delong_test(y, other, s)
    assert ab.z == pytest.approx(-ba.z)
    assert ab.pvalue == pytest.approx(ba.pvalue)


def test_needs_two_of_each_class() -> None:
    with pytest.raises(ValueError, match="two rows of each class"):
        delong_ci([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])
