"""Confidence intervals and paired tests for ML evaluation metrics.

.. include:: ../../README.md
   :start-line: 2
"""

from metricci._bootstrap import BootstrapWarning, CIResult, bootstrap_ci
from metricci.compare import PermutationResult, paired_diff_ci, paired_permutation_test
from metricci.delong import DeLongTest, delong_ci, delong_test
from metricci.intervals import auc_ci, f1_ci, hanley_mcneil_ci, lift_ci
from metricci.metrics import f1, lift_at_k, roc_auc

__all__ = [
    "BootstrapWarning",
    "CIResult",
    "DeLongTest",
    "PermutationResult",
    "auc_ci",
    "bootstrap_ci",
    "delong_ci",
    "delong_test",
    "f1",
    "f1_ci",
    "hanley_mcneil_ci",
    "lift_at_k",
    "lift_ci",
    "paired_diff_ci",
    "paired_permutation_test",
    "roc_auc",
]
__version__ = "0.1.0"
