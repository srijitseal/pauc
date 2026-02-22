"""
pAUC: A Python library for ROC curve analysis and comparison.
"""

__version__ = "0.3.0"

from .roc_core import ROC, MultiClassROC
from .stats import (
    var,
    cov,
    ci_auc,
    compare,
    ci_sensitivity,
    ci_specificity,
    venkatraman_test,
    test_operating_point,
)
from .smoothing import smooth
from .plot import plot_roc
from .power import power_roc, sample_size_roc

__all__ = [
    "ROC",
    "MultiClassROC",
    "var",
    "cov",
    "ci_auc",
    "compare",
    "smooth",
    "plot_roc",
    "ci_sensitivity",
    "ci_specificity",
    "venkatraman_test",
    "test_operating_point",
    "power_roc",
    "sample_size_roc",
]
