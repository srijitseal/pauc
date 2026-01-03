"""
pAUC: A Python library for ROC curve analysis and comparison.
"""

__version__ = "0.2.0"

from .roc_core import ROC, MultiClassROC
from .stats import compare, ci_auc, ci_thresholds, var, cov, ci_sensitivity, ci_specificity
from .plot import plot_roc
from .smoothing import smooth

__all__ = [
    "ROC",
    "MultiClassROC",
    "compare",
    "ci_auc",
    "ci_thresholds",
    "ci_specificity",
    "ci_sensitivity",
    "var",
    "cov",
    "plot_roc",
    "smooth",
]
