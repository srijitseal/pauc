import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
from numpy import trapz


# from https://github.com/PatWalters/comparing_classifiers/blob/master/delong_ci.py
# from https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return float(np.log10(2) + stats.norm.logsf(z, loc=0, scale=1).item() / np.log(10))



def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, _ = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def roc_auc_ci_score(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)

    # Handle edge cases when auc_std is zero or very small
    if auc_std < 1e-10:
        if auc == 1.0:
            ci = np.array([1.0, 1.0])
        elif auc == 0.0:
            ci = np.array([0.0, 0.0])
        else:
            # If std dev is extremely low but AUC is not exactly 0 or 1
            ci = np.array([auc, auc])
    else:
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        ci = stats.norm.ppf(
            lower_upper_q,
            loc=auc,
            scale=auc_std)

        # Ensure confidence intervals within [0,1]
        ci[ci > 1] = 1
        ci[ci < 0] = 0

    return auc, ci

def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]
        aucs.append(roc_auc_score(y_true_boot, y_score_boot))

    print("This gives an empirical confidence interval of the AUC using bootstrapping. It may differ slightly due to randomness.")
    
    aucs = np.array(aucs)
    return np.mean(aucs), np.percentile(aucs, [2.5, 97.5])
    
def bootstrap_roc_curve_ci(y_true, y_score, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    tpr_list = []
    fpr_linspace = np.linspace(0, 1, 100)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]

        fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
        tpr_interp = np.interp(fpr_linspace, fpr_boot, tpr_boot)
        tpr_interp[0] = 0.0
        tpr_list.append(tpr_interp)

    tpr_arr = np.array(tpr_list)
    tpr_mean = np.mean(tpr_arr, axis=0)
    tpr_lower = np.percentile(tpr_arr, 2.5, axis=0)
    tpr_upper = np.percentile(tpr_arr, 97.5, axis=0)

    return fpr_linspace, tpr_mean, tpr_lower, tpr_upper

def plot_roc_with_ci(y_true, y_score):
    fpr, tpr_mean, tpr_lower, tpr_upper = bootstrap_roc_curve_ci(y_true, y_score)
    auc, ci = roc_auc_ci_score(y_true, y_score)

    # AUC for the mean ROC curve
    auc_mean = trapz(tpr_mean, fpr)
    
    # AUC for the lower CI bound
    auc_lower = trapz(tpr_lower, fpr)
    
    # AUC for the upper CI bound
    auc_upper = trapz(tpr_upper, fpr)
    
   
    print(f"AUC from TPR curves: {auc_mean:.3f}")
    print(f"TPR Envelope AUC range: ({auc_lower:.3f}, {auc_upper:.3f})")
    print("TPR Envelope AUC range is not the statistical confidence interval, it's just the area under the lower/upper percentile ROC curves.")


    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # ROC curve and CI band
    ax.plot(fpr, tpr_mean, color='blue', label='Mean ROC')
    ax.fill_between(fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label='95% CI band')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)

    # Style
    ax.set_title(f"ROC Curve (AUC = {auc:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}])")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add axis arrows
    ax.annotate('', xy=(1.02, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5), xycoords='axes fraction')
    ax.annotate('', xy=(0, 1.02), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', lw=1.5), xycoords='axes fraction')

    plt.tight_layout()
    plt.show()

