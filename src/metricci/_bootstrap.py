"""Nonparametric bootstrap confidence intervals: percentile, basic and BCa.

The BCa ("bias-corrected and accelerated") interval follows Efron (1987),
"Better Bootstrap Confidence Intervals", JASA 82(397), with the acceleration
estimated by the jackknife (Efron & Tibshirani, *An Introduction to the
Bootstrap*, 1993, eq. 14.15).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

Method = Literal["percentile", "basic", "bca"]
FloatArray = NDArray[np.float64]
Statistic = Callable[..., float | FloatArray]

_METHODS: tuple[str, ...] = ("percentile", "basic", "bca")


@dataclass(frozen=True)
class CIResult:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n_resamples: int

    def __str__(self) -> str:
        pct = round(100 * self.confidence)
        return f"{self.estimate:.4f} ({pct}% CI {self.low:.4f}-{self.high:.4f}, {self.method})"


def _as_arrays(arrays: Sequence[ArrayLike]) -> list[NDArray[np.generic]]:
    if not arrays:
        raise ValueError("at least one data array is required")
    out = [np.asarray(a) for a in arrays]
    n = out[0].shape[0] if out[0].ndim else 0
    for a in out:
        if a.ndim != 1:
            raise ValueError("data arrays must be one-dimensional")
        if a.shape[0] != n:
            raise ValueError("data arrays must all have the same length")
    if n < 2:
        raise ValueError("at least two observations are required")
    return out


def _resample_indices(
    n: int,
    n_resamples: int,
    rng: np.random.Generator,
    strata: NDArray[np.generic] | None,
) -> NDArray[np.intp]:
    """Return an (n_resamples, n) matrix of row indices.

    With ``strata`` the resample draws, within each stratum, as many rows as the
    stratum holds, so stratum sizes are fixed across resamples.
    """
    if strata is None:
        return rng.integers(0, n, size=(n_resamples, n))
    idx = np.empty((n_resamples, n), dtype=np.intp)
    col = 0
    for level in np.unique(strata):
        members = np.flatnonzero(strata == level)
        k = members.shape[0]
        idx[:, col : col + k] = members[rng.integers(0, k, size=(n_resamples, k))]
        col += k
    return idx


def _evaluate(
    statistic: Statistic,
    arrays: list[NDArray[np.generic]],
    idx: NDArray[np.intp],
    vectorized: bool,
    batch: int,
) -> FloatArray:
    if vectorized:
        parts = [
            np.asarray(statistic(*(a[idx[s : s + batch]] for a in arrays)), dtype=np.float64)
            for s in range(0, idx.shape[0], batch)
        ]
        return np.concatenate(parts)
    return np.array([float(statistic(*(a[row] for a in arrays))) for row in idx])


def _jackknife(
    statistic: Statistic, arrays: list[NDArray[np.generic]], vectorized: bool, batch: int
) -> FloatArray:
    # Leave-one-out index rows are built a batch at a time: the full (n, n-1)
    # matrix is 80 MB at n = 3,000 and grows quadratically.
    n = arrays[0].shape[0]
    base = np.arange(n - 1)
    parts = []
    for s in range(0, n, batch):
        left_out = np.arange(s, min(s + batch, n))[:, None]
        idx = base[None, :] + (base[None, :] >= left_out)
        parts.append(_evaluate(statistic, arrays, idx, vectorized, batch))
    return np.concatenate(parts)


def bca_acceleration(jackknife_values: ArrayLike) -> float:
    """Jackknife estimate of the BCa acceleration constant ``a``."""
    theta = np.asarray(jackknife_values, dtype=np.float64)
    d = theta.mean() - theta
    denom = 6.0 * np.sum(d**2) ** 1.5
    if denom == 0.0:
        return 0.0
    return float(np.sum(d**3) / denom)


def bca_bias_correction(theta_hat: float, boot: ArrayLike) -> float:
    """Bias-correction ``z0``.

    Ties with the point estimate count one half. For a continuous statistic this
    is the textbook ``Phi^-1(#{theta* < theta} / B)``; for discrete statistics
    such as F1 or accuracy, where many resamples tie the estimate exactly,
    counting ties as "below" (or ignoring them) biases ``z0`` by up to the tie
    fraction.
    """
    b = np.asarray(boot, dtype=np.float64)
    p = (np.sum(b < theta_hat) + 0.5 * np.sum(b == theta_hat)) / b.shape[0]
    p = min(max(p, 0.5 / b.shape[0]), 1.0 - 0.5 / b.shape[0])
    return float(ndtri(p))


def bca_quantiles(z0: float, a: float, alpha: float) -> tuple[float, float]:
    """Adjusted lower/upper quantile levels for a ``1 - alpha`` BCa interval."""
    z = ndtri(np.array([alpha / 2, 1 - alpha / 2]))
    adj = ndtr(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return float(adj[0]), float(adj[1])


def bootstrap_ci(
    statistic: Statistic,
    *arrays: ArrayLike,
    method: Method = "bca",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = None,
    strata: ArrayLike | None = None,
    vectorized: bool = False,
    batch: int = 500,
) -> CIResult:
    """Bootstrap confidence interval for ``statistic(*arrays)``.

    Rows of all ``arrays`` are resampled jointly (paired), which is what a
    metric of ``(y_true, y_score)`` needs.

    Parameters
    ----------
    statistic
        Function of the data arrays returning a scalar. With ``vectorized=True``
        it receives arrays of shape ``(k, n)`` and must return shape ``(k,)``.
    method
        ``"percentile"``, ``"basic"`` or ``"bca"`` (default).
    strata
        Optional labels; resampling is done within each stratum. Pass the class
        labels for classification metrics so that every resample has the same
        class counts as the data and no resample is missing a class.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least 2")
    data = _as_arrays(arrays)
    n = data[0].shape[0]
    strata_arr = None if strata is None else np.asarray(strata)
    if strata_arr is not None and strata_arr.shape != (n,):
        raise ValueError("strata must have one label per observation")
    rng = np.random.default_rng(random_state)

    theta_hat = (
        float(np.asarray(statistic(*(a[None, :] for a in data)))[0])
        if vectorized
        else float(statistic(*data))
    )
    idx = _resample_indices(n, n_resamples, rng, strata_arr)
    boot = _evaluate(statistic, data, idx, vectorized, batch)
    if not np.all(np.isfinite(boot)):
        raise ValueError(
            "statistic was not finite on some resamples; for classification metrics "
            "pass strata=y_true so every resample contains both classes"
        )

    alpha = 1.0 - confidence
    if np.all(boot == boot[0]):
        low = high = float(boot[0])
    elif method == "percentile":
        low, high = (float(q) for q in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    elif method == "basic":
        ql, qh = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        low, high = 2 * theta_hat - float(qh), 2 * theta_hat - float(ql)
    else:
        z0 = bca_bias_correction(theta_hat, boot)
        acc = bca_acceleration(_jackknife(statistic, data, vectorized, batch))
        pl, ph = bca_quantiles(z0, acc, alpha)
        low, high = (float(q) for q in np.quantile(boot, [pl, ph]))
    return CIResult(theta_hat, low, high, confidence, method, n_resamples)
