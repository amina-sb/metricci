"""DeLong's nonparametric variance for ROC AUC, for one model or two paired models.

DeLong, E. R., DeLong, D. M. & Clarke-Pearson, D. L. (1988). "Comparing the
areas under two or more correlated receiver operating characteristic curves:
a nonparametric approach." *Biometrics* 44(3), 837-845.

The structural components are computed from mid-ranks as in Sun, X. & Xu, W.
(2014), "Fast implementation of DeLong's algorithm for comparing the areas
under correlated receiver operating characteristic curves", *IEEE Signal
Processing Letters* 21(11), 1389-1393: O(n log n) instead of the O(n1 * n0)
pairwise definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from metricci._bootstrap import CIResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DeLongTest:
    """Paired DeLong z-test for ``AUC(a) - AUC(b)``."""

    difference: float
    se: float
    z: float
    pvalue: float


def _components(y_true: ArrayLike, y_score: ArrayLike) -> tuple[float, FloatArray, FloatArray]:
    """AUC and the structural components V10 (one per positive) and V01 (one per
    negative); each is the AUC of that row against every row of the other class,
    with ties counting one half."""
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D and the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 < 2 or n0 < 2:
        raise ValueError("DeLong needs at least two rows of each class")
    r_all = rankdata(s)
    v10 = (r_all[pos] - rankdata(s[pos])) / n0
    v01 = 1.0 - (r_all[~pos] - rankdata(s[~pos])) / n1
    return float(v10.mean()), v10, v01


def delong_ci(y_true: ArrayLike, y_score: ArrayLike, confidence: float = 0.95) -> CIResult:
    """Wald interval for AUC with DeLong's variance, clipped to [0, 1]."""
    auc, v10, v01 = _components(y_true, y_score)
    var = np.var(v10, ddof=1) / v10.shape[0] + np.var(v01, ddof=1) / v01.shape[0]
    half = float(ndtri(0.5 + confidence / 2)) * math.sqrt(float(var))
    return CIResult(auc, max(0.0, auc - half), min(1.0, auc + half), confidence, "delong", 0)


def delong_test(y_true: ArrayLike, score_a: ArrayLike, score_b: ArrayLike) -> DeLongTest:
    """Two-sided paired test of equal AUC for two models scored on the same rows."""
    auc_a, a10, a01 = _components(y_true, score_a)
    auc_b, b10, b01 = _components(y_true, score_b)
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    cov = s10 / a10.shape[0] + s01 / a01.shape[0]
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    diff = auc_a - auc_b
    if var <= 0.0:
        return DeLongTest(diff, 0.0, 0.0 if diff == 0 else math.copysign(math.inf, diff),
                          1.0 if diff == 0 else 0.0)  # fmt: skip
    se = math.sqrt(var)
    z = diff / se
    return DeLongTest(diff, se, z, float(2 * ndtr(-abs(z))))
