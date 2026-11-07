from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats

from metricci import bootstrap_ci
from metricci._bootstrap import Method, bca_acceleration, bca_bias_correction, bca_quantiles


def _mean(x: NDArray[np.float64]) -> float:
    return float(np.mean(x))


def _mean_v(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(np.mean(x, axis=-1), dtype=np.float64)


@pytest.fixture
def skewed() -> NDArray[np.float64]:
    return np.random.default_rng(7).exponential(scale=2.0, size=60)


@pytest.mark.parametrize("method", ["percentile", "basic", "bca"])
def test_agrees_with_scipy_reference(method: Method, skewed: NDArray[np.float64]) -> None:
    # Both use 40,000 resamples, so Monte Carlo error in each endpoint is ~0.01.
    ours = bootstrap_ci(
        _mean_v, skewed, method=method, n_resamples=40_000, random_state=1, vectorized=True
    )
    ref = stats.bootstrap(
        (skewed,), np.mean, method=method, n_resamples=40_000, random_state=2, vectorized=True
    ).confidence_interval
    assert ours.low == pytest.approx(ref.low, abs=0.03)
    assert ours.high == pytest.approx(ref.high, abs=0.03)


def test_bca_shifts_right_for_right_skewed_mean(skewed: NDArray[np.float64]) -> None:
    # For a right-skewed sample the mean's sampling distribution is right-skewed
    # and BCa moves both endpoints up relative to the percentile interval.
    p = bootstrap_ci(_mean_v, skewed, method="percentile", random_state=0, vectorized=True)
    b = bootstrap_ci(_mean_v, skewed, method="bca", random_state=0, vectorized=True)
    assert b.low > p.low
    assert b.high > p.high


def test_vectorized_and_scalar_paths_agree(skewed: NDArray[np.float64]) -> None:
    a = bootstrap_ci(_mean, skewed, random_state=3, n_resamples=500)
    b = bootstrap_ci(_mean_v, skewed, random_state=3, n_resamples=500, vectorized=True)
    assert (a.low, a.high) == pytest.approx((b.low, b.high), rel=1e-12)


def test_seeded_runs_are_reproducible(skewed: NDArray[np.float64]) -> None:
    a = bootstrap_ci(_mean_v, skewed, random_state=11, vectorized=True)
    b = bootstrap_ci(_mean_v, skewed, random_state=11, vectorized=True)
    assert a == b


def test_acceleration_of_mean_matches_closed_form(skewed: NDArray[np.float64]) -> None:
    # For the sample mean the jackknife acceleration reduces to
    # sum((x - xbar)^3) / (6 * sum((x - xbar)^2) ** 1.5)  (Efron & Tibshirani, 14.15).
    n = skewed.shape[0]
    jack = (skewed.sum() - skewed) / (n - 1)
    d = skewed - skewed.mean()
    expected = np.sum(d**3) / (6 * np.sum(d**2) ** 1.5)
    assert bca_acceleration(jack) == pytest.approx(expected, rel=1e-9)


def test_bias_correction_counts_ties_as_half() -> None:
    boot = np.array([0.0, 1.0, 1.0, 1.0, 2.0])
    # 1 below, 3 tied: p = (1 + 1.5) / 5 = 0.5 -> z0 = 0
    assert bca_bias_correction(1.0, boot) == pytest.approx(0.0, abs=1e-12)


def test_bca_quantiles_reduce_to_percentile_without_correction() -> None:
    lo, hi = bca_quantiles(0.0, 0.0, 0.05)
    assert (lo, hi) == pytest.approx((0.025, 0.975), abs=1e-12)


def test_strata_fix_class_counts() -> None:
    y = np.array([0] * 30 + [1] * 5)
    seen: list[int] = []

    def count_pos(t: NDArray[np.int64]) -> float:
        seen.append(int(t.sum()))
        return float(t.mean())

    bootstrap_ci(count_pos, y, strata=y, method="percentile", n_resamples=200, random_state=0)
    assert set(seen) == {5}


def test_constant_statistic_gives_degenerate_interval() -> None:
    res = bootstrap_ci(_mean_v, np.ones(20), vectorized=True, random_state=0)
    assert res.low == res.high == 1.0


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        ({"method": "student"}, "method"),
        ({"confidence": 1.0}, "confidence"),
        ({"n_resamples": 1}, "n_resamples"),
    ],
)
def test_rejects_bad_arguments(kwargs: dict[str, object], msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        bootstrap_ci(_mean, np.arange(10.0), **kwargs)  # type: ignore[arg-type]


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci(lambda a, b: 0.0, np.arange(5), np.arange(6))


def test_nonfinite_statistic_points_to_strata() -> None:
    y = np.array([0] * 18 + [1] * 2)

    def needs_both(t: NDArray[np.int64]) -> float:
        return float("nan") if t.sum() == 0 else 1.0

    with pytest.raises(ValueError, match="strata"):
        bootstrap_ci(needs_both, y, n_resamples=500, random_state=0)
