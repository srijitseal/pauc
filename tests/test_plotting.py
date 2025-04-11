import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend for CI environments
import matplotlib.pyplot as plt
from pauc import plot_roc_with_ci

def test_plot_roc_with_ci_runs():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.75, 0.2])

    try:
        plot_roc_with_ci(y_true, y_score)
    except Exception as e:
        assert False, f"plot_roc_with_ci raised an error: {e}"
