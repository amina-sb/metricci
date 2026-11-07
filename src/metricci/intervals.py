"""One-call confidence intervals for common classification metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri

from metricci._bootstrap import CIResult, FloatArray, Method, bootstrap_ci
from metricci.metrics import f1, lift_at_k, roc_auc

Seed = int | np.random.Generator | None


def auc_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    method: Method = "bca",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: Seed = None,
) -> CIResult:
    """Bootstrap CI for ROC AUC, resampling within each class.

    Stratifying by the label keeps the class counts of every resample equal to
    the data's, so no resample lacks a class and the interval describes the AUC
    for a test set of this composition.
    """
    y = np.asarray(y_true)
    return bootstrap_ci(
        roc_auc, y, y_score, method=method, n_resamples=n_resamples, confidence=confidence,
        random_state=random_state, strata=y, vectorized=True,
    )  # fmt: skip


def f1_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    method: Method = "bca",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: Seed = None,
) -> CIResult:
    """Bootstrap CI for the F1 score of hard predictions, stratified by class."""
    y = np.asarray(y_true)
    return bootstrap_ci(
        f1, y, y_pred, method=method, n_resamples=n_resamples, confidence=confidence,
        random_state=random_state, strata=y, vectorized=True,
    )  # fmt: skip


def lift_ci(
    y_true: ArrayLike,
    y_score: ArrayLike,
    k: float = 0.1,
    *,
    method: Method = "bca",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: Seed = None,
) -> CIResult:
    """Bootstrap CI for lift in the top fraction ``k``, stratified by class."""
    y = np.asarray(y_true)

    def stat(t: FloatArray, s: FloatArray) -> float | FloatArray:
        return lift_at_k(t, s, k)

    return bootstrap_ci(
        stat, y, y_score, method=method, n_resamples=n_resamples,
        confidence=confidence, random_state=random_state, strata=y, vectorized=True,
    )  # fmt: skip


def hanley_mcneil_ci(y_true: ArrayLike, y_score: ArrayLike, confidence: float = 0.95) -> CIResult:
    """Wald interval for AUC with the Hanley & McNeil (1982) standard error.

    Hanley, J. A. & McNeil, B. J. (1982). "The meaning and use of the area under
    a receiver operating characteristic (ROC) curve." *Radiology* 143(1), 29-36,
    eq. 2. The variance assumes exponential score distributions; it is included
    as the closed-form baseline the bootstrap intervals are compared against,
    not as a recommendation. The interval is clipped to [0, 1].
    """
    y = np.asarray(y_true)
    a = float(roc_auc(y, y_score))
    n1 = int(y.sum())
    n0 = int(y.shape[0] - n1)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    z = float(ndtri(0.5 + confidence / 2))
    half = z * math.sqrt(max(var, 0.0))
    return CIResult(a, max(0.0, a - half), min(1.0, a + half), confidence, "hanley-mcneil", 0)
