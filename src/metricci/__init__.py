"""Confidence intervals and paired tests for ML evaluation metrics."""

from metricci._bootstrap import CIResult, bootstrap_ci
from metricci.metrics import f1, lift_at_k, roc_auc

__all__ = ["CIResult", "bootstrap_ci", "f1", "lift_at_k", "roc_auc"]
__version__ = "0.1.0"
