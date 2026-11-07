from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from metricci import auc_ci, f1, roc_auc
from metricci.compare import paired_diff_ci, paired_permutation_test


def _two_models(
    rng: np.random.Generator, n_pos: int = 60, n_neg: int = 140, gain: float = 0.0
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """Two models that share most of their noise, as models trained on the same
    features do; model A's signal is larger by ``gain``."""
    y = np.r_[np.ones(n_pos, dtype=np.int64), np.zeros(n_neg, dtype=np.int64)]
    shared = rng.normal(0, 1, y.shape[0])
    a = y * (1.0 + gain) + shared + rng.normal(0, 0.5, y.shape[0])
    b = y * 1.0 + shared + rng.normal(0, 0.5, y.shape[0])
    return y, a, b


def test_paired_interval_is_narrower_than_either_marginal() -> None:
    y, a, b = _two_models(np.random.default_rng(0))
    diff = paired_diff_ci(roc_auc, y, a, b, random_state=0)
    ia = auc_ci(y, a, random_state=0)
    ib = auc_ci(y, b, random_state=0)
    assert diff.estimate == pytest.approx(ia.estimate - ib.estimate, abs=1e-12)
    assert diff.high - diff.low < min(ia.high - ia.low, ib.high - ib.low)


def test_permutation_pvalue_matches_scipy() -> None:
    y, a, b = _two_models(np.random.default_rng(1), gain=0.3)
    ra = stats.rankdata(a) / a.shape[0]
    rb = stats.rankdata(b) / b.shape[0]

    def stat(x: NDArray[np.float64], z: NDArray[np.float64], axis: int = -1) -> object:
        yy = np.broadcast_to(y, x.shape)
        return np.asarray(roc_auc(yy, x)) - np.asarray(roc_auc(yy, z))

    ref = stats.permutation_test(
        (ra, rb), stat, permutation_type="samples", n_resamples=19_999,
        vectorized=True, random_state=3,
    )  # fmt: skip
    ours = paired_permutation_test(roc_auc, y, a, b, n_permutations=19_999, random_state=4)
    assert ours.statistic == pytest.approx(float(ref.statistic), abs=1e-12)
    assert ours.pvalue == pytest.approx(float(ref.pvalue), abs=0.01)


def test_identical_models_give_pvalue_one() -> None:
    y, a, _ = _two_models(np.random.default_rng(2))
    res = paired_permutation_test(roc_auc, y, a, a.copy(), n_permutations=199, random_state=0)
    assert res.statistic == 0.0
    assert res.pvalue == 1.0


def test_pvalue_is_never_zero() -> None:
    y, a, b = _two_models(np.random.default_rng(3), n_pos=200, n_neg=200, gain=3.0)
    res = paired_permutation_test(roc_auc, y, a, b, n_permutations=99, random_state=0)
    assert res.pvalue == pytest.approx(1 / 100)


def test_one_sided_alternatives_are_complementary_in_direction() -> None:
    y, a, b = _two_models(np.random.default_rng(4), gain=0.6)
    g = paired_permutation_test(roc_auc, y, a, b, alternative="greater", random_state=0)
    s = paired_permutation_test(roc_auc, y, a, b, alternative="less", random_state=0)
    assert g.pvalue < 0.05 < s.pvalue


def test_hard_predictions_are_not_rank_transformed() -> None:
    y, a, b = _two_models(np.random.default_rng(5), gain=0.5)
    pa, pb = (a > 0.5).astype(int), (b > 0.5).astype(int)
    res = paired_permutation_test(f1, y, pa, pb, n_permutations=199, random_state=0)
    assert res.statistic == pytest.approx(float(f1(y, pa)) - float(f1(y, pb)), abs=1e-12)


@pytest.mark.slow
def test_type_one_error_is_nominal_under_the_null() -> None:
    # Equal-quality models: a 0.05-level test should reject ~5% of the time.
    # 300 replications -> standard error 0.0126; band [0.015, 0.09].
    rng = np.random.default_rng(7)
    rejections = 0
    reps = 300
    for _ in range(reps):
        y, a, b = _two_models(rng)
        res = paired_permutation_test(roc_auc, y, a, b, n_permutations=499, random_state=rng)
        rejections += res.pvalue <= 0.05
    assert 0.015 <= rejections / reps <= 0.09, rejections / reps


def test_rejects_bad_alternative() -> None:
    with pytest.raises(ValueError, match="alternative"):
        paired_permutation_test(roc_auc, [0, 1], [0.1, 0.2], [0.2, 0.1], alternative="both")  # type: ignore[arg-type]
