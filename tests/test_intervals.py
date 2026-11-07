from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from metricci import auc_ci, f1_ci, hanley_mcneil_ci, lift_ci
from metricci._bootstrap import Method


def _binormal(
    rng: np.random.Generator, n_pos: int, n_neg: int, d: float
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    s = np.r_[rng.normal(d, 1, n_pos), rng.normal(0, 1, n_neg)]
    return y, s


def test_auc_ci_brackets_estimate_on_typical_data() -> None:
    y, s = _binormal(np.random.default_rng(0), 60, 140, 1.0)
    res = auc_ci(y, s, random_state=0)
    assert res.low < res.estimate < res.high
    assert res.method == "bca"
    assert 0.0 <= res.low and res.high <= 1.0


def test_f1_and_lift_helpers_run_end_to_end() -> None:
    y, s = _binormal(np.random.default_rng(1), 40, 160, 1.5)
    f = f1_ci(y, (s > 0.75).astype(int), random_state=0)
    lft = lift_ci(y, s, 0.1, random_state=0)
    assert f.low <= f.estimate <= f.high
    assert lft.low <= lft.estimate <= lft.high
    assert lft.estimate > 1.0


def test_hanley_mcneil_hand_computed_value() -> None:
    # All scores tied -> AUC = 0.5, Q1 = Q2 = 1/3. With n1 = n0 = 10:
    # var = (0.25 + 9 * (1/12) + 9 * (1/12)) / 100 = 0.0175, se = 0.132288,
    # half-width = 1.959964 * 0.132288 = 0.259279.
    y = np.r_[np.ones(10, dtype=int), np.zeros(10, dtype=int)]
    res = hanley_mcneil_ci(y, np.zeros(20))
    assert res.estimate == 0.5
    assert res.low == pytest.approx(0.5 - 0.259279, abs=1e-6)
    assert res.high == pytest.approx(0.5 + 0.259279, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_auc_interval_coverage_by_simulation(method: Method) -> None:
    """A nominal 95% interval must cover the true AUC about 95% of the time.

    Binormal scores with separation d = 1 have true AUC Phi(1 / sqrt 2) = 0.7602.
    With 300 replications the coverage estimate has standard error ~0.013, so the
    band [0.91, 0.985] fails a correct method well under 1% of the time.
    """
    true_auc = float(norm.cdf(1 / np.sqrt(2)))
    rng = np.random.default_rng(2024)
    hits = 0
    reps = 300
    for _ in range(reps):
        y, s = _binormal(rng, 50, 150, 1.0)
        res = auc_ci(y, s, method=method, n_resamples=999, random_state=rng)
        hits += res.low <= true_auc <= res.high
    coverage = hits / reps
    assert 0.91 <= coverage <= 0.985, coverage
