import numpy as np
import scipy.stats
from .roc_core import SmoothedROC


def smooth(roc, method="poly", degree=3):
    """Smooths a ROC curve."""
    if method == "poly":
        coeffs = np.polyfit(roc.fpr, roc.tpr, degree)
        smoothed_fpr = np.linspace(0, 1, 500)
        smoothed_tpr = np.clip(np.polyval(coeffs, smoothed_fpr), 0, 1)
        return SmoothedROC(roc, smoothed_fpr, smoothed_tpr)

    elif method == "binormal":
        tpr_no_ext = roc.tpr[
            (roc.tpr > 0) & (roc.tpr < 1) & (roc.fpr > 0) & (roc.fpr < 1)
        ]
        fpr_no_ext = roc.fpr[
            (roc.tpr > 0) & (roc.tpr < 1) & (roc.fpr > 0) & (roc.fpr < 1)
        ]

        if len(tpr_no_ext) < 2:
            raise ValueError("Not enough points to fit a binormal model.")

        probit_tpr = scipy.stats.norm.ppf(tpr_no_ext)
        probit_fpr = scipy.stats.norm.ppf(fpr_no_ext)

        slope, intercept, _, _, _ = scipy.stats.linregress(probit_fpr, probit_tpr)

        smoothed_fpr_probit = np.linspace(-3, 3, 500)
        smoothed_tpr_probit = intercept + slope * smoothed_fpr_probit

        smoothed_fpr_final = scipy.stats.norm.cdf(smoothed_fpr_probit)
        smoothed_tpr_final = scipy.stats.norm.cdf(smoothed_tpr_probit)
        return SmoothedROC(roc, smoothed_fpr_final, smoothed_tpr_final)
    else:
        raise NotImplementedError("Only 'poly' and 'binormal' methods are supported.")
