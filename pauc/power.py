import numpy as np
import scipy.stats
from scipy.optimize import brentq


def _variance_obuchowski(auc, n_cases, n_controls):
    q1 = auc / (2 - auc)
    q2 = (2 * auc**2) / (1 + auc)
    return (
        auc * (1 - auc)
        + (n_cases - 1) * (q1 - auc**2)
        + (n_controls - 1) * (q2 - auc**2)
    ) / (n_cases * n_controls)


def power_roc(auc, n_cases, n_controls, alpha=0.05, two_sided=True):
    var_1 = _variance_obuchowski(auc, n_cases, n_controls)
    var_0 = _variance_obuchowski(0.5, n_cases, n_controls)

    if var_0 <= 0 or var_1 <= 0:
        return 0.0

    z_alpha = (
        scipy.stats.norm.ppf(1 - alpha / 2)
        if two_sided
        else scipy.stats.norm.ppf(1 - alpha)
    )
    z_power = (np.abs(auc - 0.5) - z_alpha * np.sqrt(var_0)) / np.sqrt(var_1)
    return scipy.stats.norm.cdf(z_power)


def sample_size_roc(auc, power=0.8, kappa=1.0, alpha=0.05, two_sided=True):
    def objective(n):
        return power_roc(auc, n, n * kappa, alpha, two_sided) - power

    try:
        n_cases = brentq(objective, 2, 100000)
    except ValueError:
        n_cases = 100000

    return n_cases, n_cases * kappa
