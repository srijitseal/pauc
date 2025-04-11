import numpy as np
from pauc import roc_auc_ci_score

def test_roc_auc_ci_score_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)

    assert 0 <= auc <= 1
    assert lb <= auc <= ub

def test_roc_auc_ci_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.9, 0.95])

    auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)

    assert auc == 1.0
    assert lb <= auc <= ub
    assert ub <= 1.0

def test_roc_auc_ci_worst_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.9, 0.95, 0.1, 0.2])

    auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)

    assert auc == 0.0
    assert lb <= auc <= ub

def test_roc_auc_ci_all_zeros():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    try:
        roc_auc_ci_score(y_true, y_pred)
        assert False, "Should raise error due to lack of class 1"
    except AssertionError:
        pass

def test_roc_auc_ci_all_ones():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    try:
        roc_auc_ci_score(y_true, y_pred)
        assert False, "Should raise error due to lack of class 0"
    except AssertionError:
        pass

