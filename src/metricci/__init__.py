"""Confidence intervals and paired tests for ML evaluation metrics."""

from metricci._bootstrap import CIResult, bootstrap_ci
from metricci.intervals import auc_ci, f1_ci, hanley_mcneil_ci, lift_ci
from metricci.metrics import f1, lift_at_k, roc_auc

__all__ = [
    "CIResult",
    "auc_ci",
    "bootstrap_ci",
    "f1",
    "f1_ci",
    "hanley_mcneil_ci",
    "lift_at_k",
    "lift_ci",
    "roc_auc",
]
__version__ = "0.1.0"
