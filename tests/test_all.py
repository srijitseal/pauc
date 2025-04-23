"""
Comprehensive test suite for PAUC package.
Run with:
python -m pytest -q
or
python -m pytest tests/test_all.py -v
"""

import os
import tempfile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pauc import roc_auc_ci_score, plot_roc_with_ci


# Use non-interactive backend for CI environments
matplotlib.use("Agg")
RNG = np.random.default_rng(123)

print("Starting PAUC test suite...")


# Helper functions
def _tmp_png(name='out.png'):
    """Return a writable temporary .png path and make sure the directory exists."""
    tmp_dir = 'test_plots'
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"Creating output file: {os.path.join(tmp_dir, name)}")
    return os.path.join(tmp_dir, name)


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# Tests for ROC AUC CI score
def test_roc_auc_ci_score_basic():
    print("\n----- Testing ROC AUC CI score basic functionality -----")
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)
    print(f"Results - AUC: {auc:.4f}, CI: [{lb:.4f}, {ub:.4f}]")

    assert 0 <= auc <= 1
    assert lb <= auc <= ub
    print("✓ Test passed!")


def test_roc_auc_ci_perfect_classifier():
    print("\n----- Testing ROC AUC CI with perfect classifier -----")
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.9, 0.95])
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)
    print(f"Results - AUC: {auc:.4f}, CI: [{lb:.4f}, {ub:.4f}]")

    assert auc == 1.0
    assert lb <= auc <= ub
    assert ub <= 1.0
    print("✓ Test passed!")


def test_roc_auc_ci_worst_classifier():
    print("\n----- Testing ROC AUC CI with worst classifier -----")
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.9, 0.95, 0.1, 0.2])
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)
    print(f"Results - AUC: {auc:.4f}, CI: [{lb:.4f}, {ub:.4f}]")

    assert auc == 0.0
    assert lb <= auc <= ub
    print("✓ Test passed!")


def test_roc_auc_ci_all_zeros():
    print("\n----- Testing ROC AUC CI with all zeros in y_true -----")
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    try:
        roc_auc_ci_score(y_true, y_pred)
        assert False, "Should raise error due to lack of class 1"
    except AssertionError as e:
        print(f"✓ Expected error raised: {e}")


def test_roc_auc_ci_all_ones():
    print("\n----- Testing ROC AUC CI with all ones in y_true -----")
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    
    try:
        roc_auc_ci_score(y_true, y_pred)
        assert False, "Should raise error due to lack of class 0"
    except AssertionError as e:
        print(f"✓ Expected error raised: {e}")


# Tests for plot_roc_with_ci
def test_plot_roc_with_ci_runs():
    print("\n----- Testing plot_roc_with_ci function -----")
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.75, 0.2])
    print(f"y_true shape: {y_true.shape}, values: {y_true}")
    print(f"y_score shape: {y_score.shape}, values: {y_score}")

    try:
        print("Calling plot_roc_with_ci...")
        plot_roc_with_ci(y_true, y_score)
        print("✓ Function completed without errors")
    except Exception as e:
        assert False, f"plot_roc_with_ci raised an error: {e}"


# Tests for multiclass scenarios
def test_binary():
    print("\n----- Testing binary classification -----")
    n = 800
    y_true = RNG.integers(0, 2, size=n)                    # 0/1 labels
    logits = RNG.normal(size=n)                            # raw log-odds
    y_prob = 1 / (1 + np.exp(-logits))                     # sigmoid
    
    print(f"Sample size: {n}")
    print(f"y_true shape: {y_true.shape}, class distribution: {np.bincount(y_true)}")
    print(f"y_prob shape: {y_prob.shape}, range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    
    out_png = _tmp_png('binary.png')

    # should not raise
    print("Calling plot_roc_with_ci...")
    plot_roc_with_ci(y_true, y_prob, save_path=out_png,
                     fig_title="Binary ROC")

    # file created?
    assert os.path.isfile(out_png)
    print(f"✓ Output file created at {out_png}")


def test_multiclass():
    print("\n----- Testing multiclass classification -----")
    n, C = 1000, 4
    y_true = RNG.integers(0, C, size=n)                    # integers 0 … C-1
    logits = RNG.normal(size=(n, C))
    y_prob = _softmax(logits, axis=1)                      # softmaxed scores
    
    print(f"Sample size: {n}, Number of classes: {C}")
    print(f"y_true shape: {y_true.shape}, class distribution: {np.bincount(y_true)}")
    print(f"y_prob shape: {y_prob.shape}, sum along axis 1: {np.sum(y_prob, axis=1)[:5]}...")
    
    out_png = _tmp_png('multiclass.png')

    print("Calling plot_roc_with_ci...")
    plot_roc_with_ci(y_true, y_prob, save_path=out_png,
                     fig_title="Multiclass ROC curves")

    assert os.path.isfile(out_png)
    print(f"✓ Output file created at {out_png}")


def test_multilabel():
    print("\n----- Testing multilabel classification -----")
    n, C = 700, 5
    # allow 0–4 positive labels per sample
    y_true = (RNG.random(size=(n, C)) < RNG.uniform(0.1, 0.4, C)).astype(int)
    logits = RNG.normal(size=(n, C))
    y_prob = 1 / (1 + np.exp(-logits))                     # class-wise sigmoid
    
    print(f"Sample size: {n}, Number of labels: {C}")
    print(f"y_true shape: {y_true.shape}, positive labels per class: {y_true.sum(axis=0)}")
    print(f"y_prob shape: {y_prob.shape}, range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    
    out_png = _tmp_png('multilabel.png')

    print("Calling plot_roc_with_ci...")
    plot_roc_with_ci(y_true, y_prob, save_path=out_png,
                     fig_title="Multilabel ROC curves")

    assert os.path.isfile(out_png)
    print(f"✓ Output file created at {out_png}")
    
    
print("\nAll tests completed successfully!")
