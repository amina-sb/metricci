"""Paired comparison of two models evaluated on the same rows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

from metricci._bootstrap import CIResult, FloatArray, Method, bootstrap_ci

Metric = Callable[[ArrayLike, ArrayLike], "float | FloatArray"]
Alternative = Literal["two-sided", "greater", "less"]


@dataclass(frozen=True)
class PermutationResult:
    """Observed ``metric(a) - metric(b)`` and its permutation p-value."""

    statistic: float
    pvalue: float
    n_permutations: int
    alternative: str


def paired_diff_ci(
    metric: Metric,
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    *,
    method: Method = "bca",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> CIResult:
    """Bootstrap CI for ``metric(y, a) - metric(y, b)``.

    Both models are scored on the *same* resampled rows. Two separate intervals
    that overlap say little about the difference, because the two metrics are
    positively correlated through the shared test set; the paired interval
    accounts for that correlation and is usually much narrower.
    """
    y = np.asarray(y_true)

    def diff(t: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
        out: FloatArray = np.asarray(metric(t, a), dtype=np.float64) - np.asarray(
            metric(t, b), dtype=np.float64
        )
        return out

    return bootstrap_ci(
        diff, y, score_a, score_b, method=method, n_resamples=n_resamples,
        confidence=confidence, random_state=random_state, strata=y, vectorized=True,
    )  # fmt: skip


def _is_binary(x: NDArray[np.float64]) -> bool:
    return bool(np.all((x == 0) | (x == 1)))


def paired_permutation_test(
    metric: Metric,
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    *,
    n_permutations: int = 9999,
    alternative: Alternative = "two-sided",
    rank_transform: bool | None = None,
    random_state: int | np.random.Generator | None = None,
    batch: int = 1000,
) -> PermutationResult:
    """Test whether two models differ on a metric by swapping their outputs.

    Under the null hypothesis that the two models' outputs are exchangeable,
    swapping model A's and model B's output on any row leaves the joint
    distribution unchanged. Each permutation swaps every row independently with
    probability 1/2. The p-value counts the observed arrangement as one of the
    permutations, ``(1 + #extreme) / (1 + n_permutations)``, so it is never 0.

    ``rank_transform`` replaces each model's scores by their ranks scaled to
    (0, 1] before swapping. Swapping raw scores between a model that outputs
    probabilities and one that outputs margins produces mixed-scale score
    vectors, and the test then measures the mismatch of scales rather than the
    difference in ranking quality. Rank-based metrics (AUC, lift) are unchanged
    by the transform. The default ``None`` transforms unless both inputs are
    hard 0/1 predictions, where ranking would destroy the labels.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    y = np.asarray(y_true)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    if not (y.shape == a.shape == b.shape) or y.ndim != 1:
        raise ValueError("y_true, score_a and score_b must be 1-D and the same length")
    if rank_transform is None:
        rank_transform = not (_is_binary(a) and _is_binary(b))
    if rank_transform:
        a = rankdata(a) / a.shape[0]
        b = rankdata(b) / b.shape[0]
    rng = np.random.default_rng(random_state)

    observed = float(np.asarray(metric(y, a))) - float(np.asarray(metric(y, b)))
    # Tolerance so permutations that reproduce the observed value exactly (up to
    # rounding in a different summation order) count as "as extreme".
    tol = 1e-12 * max(1.0, abs(observed))
    extreme = 0
    for start in range(0, n_permutations, batch):
        k = min(batch, n_permutations - start)
        swap = rng.random((k, y.shape[0])) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        yy = np.broadcast_to(y, pa.shape)
        null = np.asarray(metric(yy, pa), dtype=np.float64) - np.asarray(metric(yy, pb))
        if alternative == "two-sided":
            extreme += int(np.sum(np.abs(null) >= abs(observed) - tol))
        elif alternative == "greater":
            extreme += int(np.sum(null >= observed - tol))
        else:
            extreme += int(np.sum(null <= observed + tol))
    pvalue = (1 + extreme) / (1 + n_permutations)
    return PermutationResult(observed, pvalue, n_permutations, alternative)
