import numpy as np
import scipy.stats
from scipy.integrate import trapezoid
from .roc_core import ROC


class ComparisonResult:
    """A container for the results of a ROC curve comparison."""

    def __init__(
        self,
        roc1,
        roc2,
        stat,
        p_value,
        method,
        conf_int=None,
        pauc_params=None,
        operating_point=None,
    ):
        (
            self.roc1,
            self.roc2,
            self.stat,
            self.p_value,
            self.method,
            self.conf_int,
            self.pauc_params,
            self.operating_point,
        ) = (roc1, roc2, stat, p_value, method, conf_int, pauc_params, operating_point)

    def __repr__(self):
        roc1_name, roc2_name = self.roc1.name or "roc1", self.roc2.name or "roc2"
        auc_type, stat_name = "AUC", "Z"

        if self.pauc_params:
            auc_type = f"pAUC({self.pauc_params['bounds'][0]}-{self.pauc_params['bounds'][1]} {self.pauc_params['focus']})"
            auc1, auc2 = self.roc1.partial_auc(
                **self.pauc_params
            ), self.roc2.partial_auc(**self.pauc_params)
        elif self.operating_point:
            op_focus = (
                "sensitivity"
                if "sensitivity" in self.operating_point
                else "specificity"
            )
            op_level = self.operating_point[op_focus]
            coord1 = self.roc1.get_coords(op_level, input=op_focus)
            coord2 = self.roc2.get_coords(op_level, input=op_focus)
            auc_type = (
                f"Specificity at {op_level*100}% Sensitivity"
                if op_focus == "sensitivity"
                else f"Sensitivity at {op_level*100}% Specificity"
            )
            auc1 = (
                coord1["specificity"]
                if op_focus == "sensitivity"
                else coord1["sensitivity"]
            )
            auc2 = (
                coord2["specificity"]
                if op_focus == "sensitivity"
                else coord2["sensitivity"]
            )
        else:
            auc1, auc2 = self.roc1.auc, self.roc2.auc

        if "Venkatraman" in self.method:
            stat_name, auc_type = "D", "Curve"

        ci_str = (
            f"95 percent confidence interval:\n {self.conf_int[0]:.3f} {self.conf_int[1]:.3f}\n"
            if self.conf_int is not None
            else ""
        )
        return (
            f"{self.method}\n\ndata:  {roc1_name} and {roc2_name}\n"
            f"{stat_name} = {self.stat:.3f}, p-value = {self.p_value:.3f}\n"
            f"alternative hypothesis: true difference in {auc_type} is not equal to 0\n{ci_str}"
            f"sample estimates:\n{auc_type} of {roc1_name} {auc_type} of {roc2_name}\n"
            f"      {auc1:.3f}       {auc2:.3f}"
        )


def _delong_variance(roc):
    """Calculates the variance of an AUC using DeLong's method."""
    if roc.n_cases < 2 or roc.n_controls < 2:
        return 0, np.array([]), np.array([])
    v10 = np.array(
        [(np.sum(roc.controls < case) / roc.n_controls) for case in roc.cases]
    )
    v01 = np.array(
        [(np.sum(roc.cases > control) / roc.n_cases) for control in roc.controls]
    )
    s10, s01 = np.var(v10, ddof=1), np.var(v01, ddof=1)
    return (s10 / roc.n_cases) + (s01 / roc.n_controls), v10, v01


def var(roc, method="delong"):
    """Public function to calculate the variance of a ROC curve's AUC."""
    if method == "delong":
        variance, _, _ = _delong_variance(roc)
        return variance
    else:
        raise NotImplementedError(
            "Only DeLong's method is currently supported for public variance calculation."
        )


def _delong_covariance(roc1, roc2):
    """Calculates covariance between two paired AUCs using DeLong's method."""
    _, v10_1, v01_1 = _delong_variance(roc1)
    _, v10_2, v01_2 = _delong_variance(roc2)
    if v10_1.size == 0 or v10_2.size == 0:
        return 0
    cov_10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    cov_01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    return (cov_10 / roc1.n_cases) + (cov_01 / roc1.n_controls)


def cov(roc1, roc2, method="delong"):
    """Public function to calculate the covariance between two paired ROC curves' AUCs."""
    if method == "delong":
        return _delong_covariance(roc1, roc2)
    else:
        raise NotImplementedError(
            "Only DeLong's method is currently supported for public covariance calculation."
        )


def compare(
    roc1,
    roc2,
    method="delong",
    paired=None,
    n_boot=2000,
    partial_auc_focus=None,
    partial_auc_bounds=None,
    sensitivity=None,
    specificity=None,
):
    """Compares two ROC curves."""
    if paired is None:
        paired = len(roc1.y_true) == len(roc2.y_true) and np.array_equal(
            roc1.y_true, roc2.y_true
        )

    pauc_params = (
        {"focus": partial_auc_focus, "bounds": partial_auc_bounds}
        if partial_auc_focus
        else None
    )

    if method in ["sensitivity", "specificity"]:
        if method == "sensitivity" and sensitivity is None:
            raise ValueError("A sensitivity level must be provided.")
        if method == "specificity" and specificity is None:
            raise ValueError("A specificity level must be provided.")

        diffs = []
        op_focus = method
        level = sensitivity if op_focus == "sensitivity" else specificity

        for _ in range(n_boot):
            idx = np.random.choice(len(roc1.y_true), len(roc1.y_true), replace=True)
            boot_roc1 = ROC(roc1.y_true[idx], roc1.y_score[idx])

            if paired:
                boot_roc2 = ROC(roc2.y_true[idx], roc2.y_score[idx])
            else:
                idx2 = np.random.choice(
                    len(roc2.y_true), len(roc2.y_true), replace=True
                )
                boot_roc2 = ROC(roc2.y_true[idx2], roc2.y_score[idx2])

            coord1 = boot_roc1.get_coords(level, input=op_focus)
            coord2 = boot_roc2.get_coords(level, input=op_focus)

            val1 = (
                coord1["specificity"]
                if op_focus == "sensitivity"
                else coord1["sensitivity"]
            )
            val2 = (
                coord2["specificity"]
                if op_focus == "sensitivity"
                else coord2["sensitivity"]
            )
            diffs.append(val1 - val2)

        observed_coord1 = roc1.get_coords(level, input=op_focus)
        observed_coord2 = roc2.get_coords(level, input=op_focus)
        observed_val1 = (
            observed_coord1["specificity"]
            if op_focus == "sensitivity"
            else observed_coord1["sensitivity"]
        )
        observed_val2 = (
            observed_coord2["specificity"]
            if op_focus == "sensitivity"
            else observed_coord2["sensitivity"]
        )
        observed_diff = observed_val1 - observed_val2

        se_diff = np.std(diffs, ddof=1)
        z = observed_diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        method_str = f"Bootstrap test for two ROC curves at {level*100}% {op_focus}"
        op_point = {op_focus: level}
        return ComparisonResult(
            roc1, roc2, z, p_value, method_str, operating_point=op_point
        )

    elif method == "delong":
        if pauc_params:
            raise ValueError("DeLong's test for partial AUC is not supported.")
        auc_diff, var1, var2 = (
            roc1.auc - roc2.auc,
            _delong_variance(roc1)[0],
            _delong_variance(roc2)[0],
        )
        if paired:
            se_diff = np.sqrt(var1 + var2 - 2 * _delong_covariance(roc1, roc2))
            method_str = "DeLong's test for two correlated ROC curves"
        else:
            se_diff, method_str = (
                np.sqrt(var1 + var2),
                "DeLong's test for two uncorrelated ROC curves",
            )
        z = auc_diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        conf_int = (
            (auc_diff - 1.96 * se_diff, auc_diff + 1.96 * se_diff)
            if se_diff > 0
            else (auc_diff, auc_diff)
        )
        return ComparisonResult(roc1, roc2, z, p_value, method_str, conf_int)

    elif method == "bootstrap":
        diffs = []
        for _ in range(n_boot):
            idx_cases = np.random.choice(roc1.n_cases, roc1.n_cases, replace=True)
            idx_controls = np.random.choice(
                roc1.n_controls, roc1.n_controls, replace=True
            )
            y_true_boot = np.concatenate(
                [np.ones(roc1.n_cases), np.zeros(roc1.n_controls)]
            )
            score1 = np.concatenate(
                [roc1.cases[idx_cases], roc1.controls[idx_controls]]
            )
            boot_roc1 = ROC(y_true_boot, score1)

            if paired:
                score2 = np.concatenate(
                    [roc2.cases[idx_cases], roc2.controls[idx_controls]]
                )
            else:
                idx_cases2 = np.random.choice(roc2.n_cases, roc2.n_cases, replace=True)
                idx_controls2 = np.random.choice(
                    roc2.n_controls, roc2.n_controls, replace=True
                )
                score2 = np.concatenate(
                    [roc2.cases[idx_cases2], roc2.controls[idx_controls2]]
                )
            boot_roc2 = ROC(y_true_boot, score2)

            auc1 = (
                boot_roc1.partial_auc(**pauc_params) if pauc_params else boot_roc1.auc
            )
            auc2 = (
                boot_roc2.partial_auc(**pauc_params) if pauc_params else boot_roc2.auc
            )
            diffs.append(auc1 - auc2)

        auc_diff = (roc1.partial_auc(**pauc_params) if pauc_params else roc1.auc) - (
            roc2.partial_auc(**pauc_params) if pauc_params else roc2.auc
        )
        se_diff = np.std(diffs, ddof=1)
        z = auc_diff / se_diff if se_diff > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
        method_str = "Bootstrap test for two ROC curves"
        return ComparisonResult(
            roc1, roc2, z, p_value, method_str, pauc_params=pauc_params
        )

    elif method == "venkatraman":
        if pauc_params:
            raise ValueError("Venkatraman's test for partial AUC is not supported.")
        observed_stat = trapezoid(
            np.abs(roc1.tpr - np.interp(roc1.fpr, roc2.fpr, roc2.tpr)), roc1.fpr
        )

        perm_stats = []
        y_true = roc1.y_true
        scores1, scores2 = roc1.y_score, roc2.y_score

        for _ in range(n_boot):
            if paired:
                swap = np.random.randint(0, 2, len(y_true)).astype(bool)
                perm_scores1, perm_scores2 = np.where(swap, scores2, scores1), np.where(
                    swap, scores1, scores2
                )
                perm_roc1, perm_roc2 = ROC(y_true, perm_scores1), ROC(
                    y_true, perm_scores2
                )
            else:
                perm_labels = np.random.permutation(y_true)
                perm_roc1, perm_roc2 = ROC(perm_labels, scores1), ROC(
                    perm_labels, scores2
                )

            perm_stat = trapezoid(
                np.abs(
                    perm_roc1.tpr
                    - np.interp(perm_roc1.fpr, perm_roc2.fpr, perm_roc2.tpr)
                ),
                perm_roc1.fpr,
            )
            perm_stats.append(perm_stat)

        p_value = np.mean(np.array(perm_stats) >= observed_stat)
        method_str = f"Venkatraman's test for two {'correlated' if paired else 'uncorrelated'} ROC curves"
        return ComparisonResult(roc1, roc2, observed_stat, p_value, method_str)
    else:
        raise ValueError(
            "Method must be one of 'delong', 'bootstrap', 'venkatraman', 'sensitivity', 'specificity'"
        )


def ci_auc(
    roc, method="delong", conf_level=0.95, n_boot=2000, bounds=None, focus="specificity"
):
    """
    Calculates the confidence interval for the AUC.

    This function handles both full AUC (method='delong' or 'bootstrap') and
    partial AUC (method='bootstrap' with 'bounds' specified).
    """
    alpha = 1 - conf_level
    if method == "delong":
        if bounds:
            raise ValueError(
                "DeLong's method does not support partial AUC confidence intervals."
            )
        variance, _, _ = _delong_variance(roc)
        se = np.sqrt(variance)
        z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
        return (roc.auc - z_crit * se, roc.auc + z_crit * se)
    elif method == "bootstrap":
        if bounds and (bounds != (0.0, 1.0)):
            pauc_params = {"focus": focus, "bounds": bounds}
            paucs = []
            for _ in range(n_boot):
                idx = np.random.choice(len(roc.y_true), len(roc.y_true), replace=True)
                boot_roc = ROC(roc.y_true[idx], roc.y_score[idx])
                paucs.append(boot_roc.partial_auc(**pauc_params))
            return (
                np.percentile(paucs, 100 * (alpha / 2)),
                np.percentile(paucs, 100 * (1 - alpha / 2)),
            )
        else:  # Full AUC
            aucs = [
                ROC(roc.y_true[idx], roc.y_score[idx]).auc
                for idx in [
                    np.random.choice(len(roc.y_true), len(roc.y_true), replace=True)
                    for _ in range(n_boot)
                ]
            ]
            return (
                np.percentile(aucs, 100 * (alpha / 2)),
                np.percentile(aucs, 100 * (1 - alpha / 2)),
            )
    else:
        raise ValueError("Method must be 'delong' or 'bootstrap'")


def _ci_coords_bootstrap(roc, points, focus, n_boot):
    bootstrapped_values = []
    for _ in range(n_boot):
        idx = np.random.choice(len(roc.y_true), len(roc.y_true), replace=True)
        boot_roc = ROC(roc.y_true[idx], roc.y_score[idx])

        if focus == "specificity":
            spec = 1 - boot_roc.fpr
            xp, fp = spec[::-1], boot_roc.tpr[::-1]
            bootstrapped_values.append(np.interp(points, xp, fp))
        else:
            xp, fp = boot_roc.tpr, 1 - boot_roc.fpr
            sort_idx = np.argsort(xp)
            bootstrapped_values.append(np.interp(points, xp[sort_idx], fp[sort_idx]))
    return np.array(bootstrapped_values)


def ci_thresholds(roc, threshold, conf_level=0.95, n_boot=2000):
    """Calculates CIs for sensitivity and specificity at a given threshold."""
    alpha = 1 - conf_level
    sens_vals, spec_vals = [], []

    if threshold == "best":
        threshold = roc.get_coords("best")["threshold"]

    for _ in range(n_boot):
        idx = np.random.choice(len(roc.y_true), len(roc.y_true), replace=True)
        boot_roc = ROC(roc.y_true[idx], roc.y_score[idx])
        coords = boot_roc.get_coords(threshold, input="threshold")
        sens_vals.append(coords["sensitivity"])
        spec_vals.append(coords["specificity"])

    ci_sens = (
        np.percentile(sens_vals, 100 * (alpha / 2)),
        np.percentile(sens_vals, 100 * (1 - alpha / 2)),
    )
    ci_spec = (
        np.percentile(spec_vals, 100 * (alpha / 2)),
        np.percentile(spec_vals, 100 * (1 - alpha / 2)),
    )
    return {"sensitivity_ci": ci_sens, "specificity_ci": ci_spec}


def ci_specificity(roc, sensitivities, conf_level=0.95, n_boot=2000):
    alpha = 1 - conf_level
    points = np.asarray(sensitivities)
    boot_values = _ci_coords_bootstrap(roc, points, "sensitivity", n_boot)
    return np.percentile(boot_values, 100 * (alpha / 2), axis=0), np.percentile(
        boot_values, 100 * (1 - alpha / 2), axis=0
    )


def ci_sensitivity(roc, specificities, conf_level=0.95, n_boot=2000):
    alpha = 1 - conf_level
    points = np.asarray(specificities)
    boot_values = _ci_coords_bootstrap(roc, points, "specificity", n_boot)
    return np.percentile(boot_values, 100 * (alpha / 2), axis=0), np.percentile(
        boot_values, 100 * (1 - alpha / 2), axis=0
    )


