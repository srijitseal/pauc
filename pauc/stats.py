import numpy as np
import scipy.stats
from .roc_core import ROC


class ComparisonResult:
    def __init__(self, stat, p_value, method, conf_int=None, diff=None):
        self.stat, self.p_value, self.method, self.conf_int, self.estimate = (
            stat,
            p_value,
            method,
            conf_int,
            diff,
        )

    def __repr__(self):
        return (
            f"\n\t{self.method}\n\n"
            f"data:  ROC curves\n"
            f"D = {self.estimate:.4f}, Z = {self.stat:.4f}, p-value = {self.p_value:.4g}\n"
            f"95 percent confidence interval:\n"
            f" {self.conf_int[0]:.4f} {self.conf_int[1]:.4f}\n"
            if self.conf_int
            else ""
        )


def _compute_mid_ranks(x):
    sorter = np.argsort(x)
    inv = np.empty(x.size, dtype=np.intp)
    inv[sorter] = np.arange(x.size, dtype=np.intp)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    lower = np.zeros_like(upper)
    lower[1:] = upper[:-1]
    return ((lower + upper + 1.0) / 2.0)[inverse]


def _delong_placements(roc):
    x, y = roc.aligned_cases, roc.aligned_controls
    r = _compute_mid_ranks(np.concatenate([x, y]))
    return (r[: roc.n_cases] - _compute_mid_ranks(x)) / roc.n_controls, (
        r[roc.n_cases :] - _compute_mid_ranks(y)
    ) / roc.n_cases


def var(roc, method="delong", n_boot=2000, stratified=True):
    if method == "delong":
        v10, v01 = _delong_placements(roc)
        return (np.var(v10, ddof=1) / roc.n_cases) + (
            np.var(v01, ddof=1) / roc.n_controls
        )
    elif method == "bootstrap":
        aucs = []
        for _ in range(n_boot):
            if stratified:
                b_cases = np.random.choice(roc.cases, roc.n_cases, replace=True)
                b_controls = np.random.choice(
                    roc.controls, roc.n_controls, replace=True
                )
                b_y = np.concatenate([np.ones(len(b_cases)), np.zeros(len(b_controls))])
                b_scores = np.concatenate([b_cases, b_controls])
            else:
                idx = np.random.choice(len(roc.y_true), len(roc.y_true), replace=True)
                b_y, b_scores = roc.y_true[idx], roc.y_score[idx]
            b_roc = ROC(
                b_y,
                b_scores,
                direction=roc.direction,
                partial_auc=roc.partial_auc_range,
                partial_auc_focus=roc.partial_auc_focus,
                standardize_pauc=roc.standardize_pauc,
            )
            aucs.append(b_roc.auc)
        return np.var(aucs, ddof=1)


def cov(roc1, roc2, method="delong"):
    if method == "delong":
        v10_1, v01_1 = _delong_placements(roc1)
        v10_2, v01_2 = _delong_placements(roc2)
        return (np.cov(v10_1, v10_2, ddof=1)[0, 1] / roc1.n_cases) + (
            np.cov(v01_1, v01_2, ddof=1)[0, 1] / roc1.n_controls
        )
    return 0.0


def compare(roc1, roc2, method="delong", paired=True, stratified=True, n_boot=2000):
    diff_obs = roc1.auc - roc2.auc
    if method == "delong":
        v1, v2 = var(roc1, "delong"), var(roc2, "delong")
        covariance = cov(roc1, roc2, "delong") if paired else 0.0
        sigma_diff = np.sqrt(max(0, v1 + v2 - 2 * covariance))
        z_score = diff_obs / sigma_diff if sigma_diff > 0 else 0
        p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))
        return ComparisonResult(
            z_score,
            p_value,
            "DeLong",
            (diff_obs - 1.96 * sigma_diff, diff_obs + 1.96 * sigma_diff),
            diff_obs,
        )

    elif method == "bootstrap":
        resampled_diffs = []
        for _ in range(n_boot):
            if paired:
                if stratified:
                    idx_c = np.random.choice(
                        np.where(roc1.y_true == roc1.cases_label)[0],
                        roc1.n_cases,
                        replace=True,
                    )
                    idx_n = np.random.choice(
                        np.where(roc1.y_true == roc1.controls_label)[0],
                        roc1.n_controls,
                        replace=True,
                    )
                    boot_idx = np.concatenate([idx_c, idx_n])
                else:
                    boot_idx = np.random.choice(
                        len(roc1.y_true), len(roc1.y_true), replace=True
                    )
                r1 = ROC(
                    roc1.y_true[boot_idx],
                    roc1.y_score[boot_idx],
                    partial_auc=roc1.partial_auc_range,
                )
                r2 = ROC(
                    roc2.y_true[boot_idx],
                    roc2.y_score[boot_idx],
                    partial_auc=roc2.partial_auc_range,
                )
            else:
                idx1 = np.random.choice(
                    len(roc1.y_true), len(roc1.y_true), replace=True
                )
                idx2 = np.random.choice(
                    len(roc2.y_true), len(roc2.y_true), replace=True
                )
                r1 = ROC(
                    roc1.y_true[idx1],
                    roc1.y_score[idx1],
                    partial_auc=roc1.partial_auc_range,
                )
                r2 = ROC(
                    roc2.y_true[idx2],
                    roc2.y_score[idx2],
                    partial_auc=roc2.partial_auc_range,
                )
            resampled_diffs.append(r1.auc - r2.auc)

        sd_diff = np.std(resampled_diffs, ddof=1)
        z_score = diff_obs / sd_diff if sd_diff > 0 else 0
        return ComparisonResult(
            z_score,
            2 * (1 - scipy.stats.norm.cdf(abs(z_score))),
            "Bootstrap",
            (
                np.percentile(resampled_diffs, 2.5),
                np.percentile(resampled_diffs, 97.5),
            ),
            diff_obs,
        )


def ci_auc(roc, conf_level=0.95, method="delong", n_boot=2000):
    alpha = 1 - conf_level
    if method == "delong":
        sd = np.sqrt(var(roc, "delong"))
        crit = scipy.stats.norm.ppf(1 - alpha / 2)
        return roc.auc - crit * sd, roc.auc + crit * sd
    elif method == "bootstrap":
        aucs = []
        for _ in range(n_boot):
            ix_cases = np.random.choice(roc.n_cases, roc.n_cases, replace=True)
            ix_controls = np.random.choice(roc.n_controls, roc.n_controls, replace=True)
            b_cases = roc.cases[ix_cases]
            b_controls = roc.controls[ix_controls]
            b_y = np.concatenate([np.ones(len(b_cases)), np.zeros(len(b_controls))])
            b_s = np.concatenate([b_cases, b_controls])
            b_roc = ROC(b_y, b_s, partial_auc=roc.partial_auc_range)
            aucs.append(b_roc.auc)
        return np.percentile(aucs, 100 * alpha / 2), np.percentile(
            aucs, 100 * (1 - alpha / 2)
        )


def _approx(x, y, xout):
    """Replicates R's approx() function with ties=mean for empirical ROC interpolation."""
    ux, indices = np.unique(x, return_inverse=True)
    uy = np.bincount(indices, weights=y) / np.bincount(indices)
    return np.interp(xout, ux, uy)


def ci_sensitivity(roc, specificities, conf_level=0.95, n_boot=2000):
    alpha = 1 - conf_level
    target_fprs = 1 - np.asarray(specificities)
    boot_tprs = []
    for _ in range(n_boot):
        ix_cases = np.random.choice(roc.n_cases, roc.n_cases, replace=True)
        ix_controls = np.random.choice(roc.n_controls, roc.n_controls, replace=True)
        b_cases = roc.cases[ix_cases]
        b_controls = roc.controls[ix_controls]
        b_y = np.concatenate([np.ones(len(b_cases)), np.zeros(len(b_controls))])
        b_s = np.concatenate([b_cases, b_controls])
        b_roc = ROC(b_y, b_s, direction=roc.direction)

        tprs = _approx(b_roc.fpr, b_roc.tpr, target_fprs)
        boot_tprs.append(tprs)

    boot_tprs = np.array(boot_tprs)
    lower = np.percentile(boot_tprs, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_tprs, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


def ci_specificity(roc, sensitivities, conf_level=0.95, n_boot=2000):
    alpha = 1 - conf_level
    target_tprs = np.asarray(sensitivities)
    boot_fprs = []
    for _ in range(n_boot):
        ix_cases = np.random.choice(roc.n_cases, roc.n_cases, replace=True)
        ix_controls = np.random.choice(roc.n_controls, roc.n_controls, replace=True)
        b_cases = roc.cases[ix_cases]
        b_controls = roc.controls[ix_controls]
        b_y = np.concatenate([np.ones(len(b_cases)), np.zeros(len(b_controls))])
        b_s = np.concatenate([b_cases, b_controls])
        b_roc = ROC(b_y, b_s, direction=roc.direction)

        fprs = _approx(b_roc.tpr, b_roc.fpr, target_tprs)
        boot_fprs.append(fprs)

    boot_fprs = np.array(boot_fprs)
    return np.percentile(1 - boot_fprs, 100 * alpha / 2, axis=0), np.percentile(
        1 - boot_fprs, 100 * (1 - alpha / 2), axis=0
    )


def venkatraman_test(roc1, roc2, n_perm=2000, paired=True):
    """Permutation test for ROC curve shape difference evaluated rigidly on the step-function grid."""

    def calc_E(r1, r2):
        fpr_grid = np.sort(np.unique(np.concatenate([r1.fpr, r2.fpr])))
        idx1 = np.clip(
            np.searchsorted(r1.fpr, fpr_grid, side="right") - 1, 0, len(r1.tpr) - 1
        )
        tpr1 = r1.tpr[idx1]

        idx2 = np.clip(
            np.searchsorted(r2.fpr, fpr_grid, side="right") - 1, 0, len(r2.tpr) - 1
        )
        tpr2 = r2.tpr[idx2]
        return np.sum(np.abs(tpr1 - tpr2))

    E_obs = calc_E(roc1, roc2)
    E_null = []

    if paired:
        n_samples = len(roc1.y_true)
        for _ in range(n_perm):
            swap = np.random.binomial(1, 0.5, size=n_samples).astype(bool)
            s1_boot = np.where(swap, roc2.y_score, roc1.y_score)
            s2_boot = np.where(swap, roc1.y_score, roc2.y_score)
            r1 = ROC(roc1.y_true, s1_boot, direction=roc1.direction)
            r2 = ROC(roc2.y_true, s2_boot, direction=roc2.direction)
            E_null.append(calc_E(r1, r2))
    else:
        y_combined = np.concatenate([roc1.y_true, roc2.y_true])
        s_combined = np.concatenate([roc1.y_score, roc2.y_score])
        n1 = len(roc1.y_true)
        for _ in range(n_perm):
            idx = np.random.permutation(len(y_combined))
            r1 = ROC(
                y_combined[idx[:n1]], s_combined[idx[:n1]], direction=roc1.direction
            )
            r2 = ROC(
                y_combined[idx[n1:]], s_combined[idx[n1:]], direction=roc2.direction
            )
            E_null.append(calc_E(r1, r2))

    p_val = np.mean(np.array(E_null) >= E_obs)
    return ComparisonResult(E_obs, p_val, "Venkatraman")


def test_operating_point(
    roc1, roc2, point, point_type="specificity", n_boot=2000, paired=True
):
    """Test difference at a specific operating point using bootstrap Z-score with exact R-style interpolation."""

    def get_stat(r):
        if point_type == "specificity":
            return _approx(r.fpr, r.tpr, 1.0 - point)
        else:
            return _approx(r.tpr, r.fpr, point)

    obs_diff = get_stat(roc1) - get_stat(roc2)
    diffs = []
    n_samples = len(roc1.y_true)

    for _ in range(n_boot):
        try:
            if paired:
                idx = np.random.choice(n_samples, n_samples, replace=True)
                b1 = ROC(roc1.y_true[idx], roc1.y_score[idx], direction=roc1.direction)
                b2 = ROC(roc2.y_true[idx], roc2.y_score[idx], direction=roc2.direction)
            else:
                idx1 = np.random.choice(n_samples, n_samples, replace=True)
                idx2 = np.random.choice(
                    len(roc2.y_true), len(roc2.y_true), replace=True
                )
                b1 = ROC(
                    roc1.y_true[idx1], roc1.y_score[idx1], direction=roc1.direction
                )
                b2 = ROC(
                    roc2.y_true[idx2], roc2.y_score[idx2], direction=roc2.direction
                )

            diffs.append(get_stat(b1) - get_stat(b2))
        except ValueError:
            # Bypass invalid resamples containing only 1 class
            continue

    sd_diff = np.std(diffs, ddof=1)
    z_score = obs_diff / sd_diff if sd_diff > 0 else 0
    p_val = 2 * (1 - scipy.stats.norm.cdf(abs(z_score)))

    return ComparisonResult(
        z_score, p_val, f"Test at {point_type}={point}", diff=obs_diff
    )
