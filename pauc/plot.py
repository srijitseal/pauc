import matplotlib.pyplot as plt
import numpy as np
from .stats import ci_sensitivity


def plot_roc(roc_objects, title="ROC Curves", plot_ci=True, annotate_best=False):
    """
    Plots one or more ROC objects on a single axis.
    """
    if not isinstance(roc_objects, list):
        roc_objects = [roc_objects]

    plt.figure(figsize=(8, 8))

    for roc in roc_objects:
        label = roc.name if roc.name else "ROC Curve"
        auc_label = f"{label} (AUC = {roc.auc:.3f})"
        (line,) = plt.plot(roc.fpr, roc.tpr, label=auc_label)

        color = line.get_color()

        if plot_ci and roc.y_true is not None:
            points = np.linspace(1, 0, 100)  # specificity points
            lower, upper = ci_sensitivity(roc, points)
            fpr_points = 1 - points
            sort_idx = np.argsort(fpr_points)
            plt.fill_between(
                fpr_points[sort_idx],
                lower[sort_idx],
                upper[sort_idx],
                color=color,
                alpha=0.2,
            )

        if annotate_best and roc.thresholds is not None:
            coords = roc.get_coords("best")
            plt.plot(
                1 - coords["specificity"],
                coords["sensitivity"],
                "o",
                color=color,
                markersize=8,
            )
            plt.annotate(
                f"Thresh={coords['threshold']:.2f}",
                (1 - coords["specificity"], coords["sensitivity"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    plt.plot(
        [0, 1], [0, 1], color="grey", linestyle=":", label="No Skill (AUC = 0.500)"
    )
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("1 - Specificity (False Positive Rate)")
    plt.ylabel("Sensitivity (True Positive Rate)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.show()
    return plt


