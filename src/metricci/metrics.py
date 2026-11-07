"""Classification metrics that work on a single sample or a stack of resamples.

Every function accepts arrays of shape ``(n,)`` and returns a float, or arrays of
shape ``(k, n)`` (one resample per row) and returns shape ``(k,)``. The stacked
form is what makes thousands of bootstrap resamples cheap.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rankdata

FloatArray = NDArray[np.float64]


def _unstack(value: FloatArray, was_1d: bool) -> float | FloatArray:
    return float(value[0]) if was_1d else value


def _prep(y_true: ArrayLike, other: ArrayLike) -> tuple[NDArray[np.bool_], FloatArray, bool]:
    t = np.asarray(y_true)
    o = np.asarray(other, dtype=np.float64)
    if t.shape != o.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {o.shape}")
    if t.ndim not in (1, 2):
        raise ValueError("inputs must be 1-D or 2-D")
    if not np.all((t == 0) | (t == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    was_1d = t.ndim == 1
    return np.atleast_2d(t).astype(bool), np.atleast_2d(o), was_1d


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float | FloatArray:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Tied scores get mid-ranks, i.e. a tied (positive, negative) pair counts one
    half, which matches :func:`sklearn.metrics.roc_auc_score`. Rows that lack
    one of the classes give ``nan``.
    """
    t, s, was_1d = _prep(y_true, y_score)
    ranks = rankdata(s, axis=-1)
    n_pos = t.sum(axis=-1)
    n_neg = t.shape[-1] - n_pos
    rank_sum = np.where(t, ranks, 0.0).sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return _unstack(np.where((n_pos > 0) & (n_neg > 0), auc, np.nan), was_1d)


def f1(y_true: ArrayLike, y_pred: ArrayLike) -> float | FloatArray:
    """F1 score of hard 0/1 predictions; 0 when there are no positives at all
    (no true positives, predicted or actual), as sklearn's ``zero_division=0``."""
    t, p_float, was_1d = _prep(y_true, y_pred)
    p = p_float.astype(bool)
    tp = np.sum(t & p, axis=-1)
    fp = np.sum(~t & p, axis=-1)
    fn = np.sum(t & ~p, axis=-1)
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0, 2 * tp / denom, 0.0)
    return _unstack(score.astype(np.float64), was_1d)


def lift_at_k(y_true: ArrayLike, y_score: ArrayLike, k: float = 0.1) -> float | FloatArray:
    """Lift in the top fraction ``k`` of rows ranked by score.

    The precision among the ``ceil(k * n)`` highest-scored rows divided by the
    overall positive rate. Scores tied across the cutoff are broken by row order
    (a stable sort), so a model that outputs many identical scores gets the lift
    of the rows that happen to come first, not an average; a reader comparing
    lift between models with coarse scores should know this.
    """
    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1]")
    t, s, was_1d = _prep(y_true, y_score)
    n = t.shape[-1]
    top = max(1, math.ceil(k * n))
    order = np.argsort(-s, axis=-1, kind="stable")[:, :top]
    hits = np.take_along_axis(t, order, axis=-1).sum(axis=-1)
    base = t.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(base > 0, (hits / top) / base, np.nan)
    return _unstack(lift.astype(np.float64), was_1d)
