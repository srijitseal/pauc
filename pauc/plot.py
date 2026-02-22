import matplotlib.pyplot as plt
import numpy as np
from .stats import ci_sensitivity, ci_specificity


def setup_tufte_style():
    """Configures global matplotlib params for Tufte-like aesthetics."""
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#333333",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_roc(
    rocs,
    ax=None,
    title=None,
    colors=None,
    show_auc=True,
    shade_auc=False,
    shade_alpha=0.05,
    plot_ci=False,
    ci_type="sensitivity",
    ci_alpha=0.15,
    annotate_best=False,
    best_method="youden",
    **kwargs,
):
    """
    Plots a Tufte-style minimalist ROC curve for one or multiple models.

    Parameters:
        rocs (ROC or list of ROC): The pAUC ROC object(s) to plot.
        ax (matplotlib.axes.Axes, optional): Existing axes to plot on.
        title (str, optional): Title for the plot.
        colors (list, optional): List of colors for multiple curves.
        show_auc (bool): Include AUC in the legend label.
        shade_auc (bool): Shade the area under the curve (or pAUC).
        shade_alpha (float): Transparency for AUC shading.
        plot_ci (bool): Shade confidence intervals.
        ci_type (str): "sensitivity" or "specificity".
        ci_alpha (float): Transparency for CI shading.
        annotate_best (bool): Highlight the optimal operating point.
        best_method (str): "youden" or "topleft".
        **kwargs: Additional arguments passed to matplotlib.pyplot.plot().

    Returns:
        matplotlib.axes.Axes: The axes object for further user customization.
    """
    setup_tufte_style()

    # Normalize input to a list
    if not isinstance(rocs, (list, tuple)):
        rocs = [rocs]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color="#cccccc", linestyle="--", linewidth=1, zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("1 - Specificity (FPR)", fontsize=11, labelpad=8)
        ax.set_ylabel("Sensitivity (TPR)", fontsize=11, labelpad=8)

        # Detach spines slightly for the Tufte "floating" look
        ax.spines["left"].set_position(("outward", 5))
        ax.spines["bottom"].set_position(("outward", 5))

    if title:
        ax.set_title(title, fontsize=12, pad=15)

    # Clean Tufte-compatible color palette
    if colors is None:
        colors = ["#111111", "#b22222", "#4682b4", "#228b22", "#8b008b", "#d2691e"]

    kwargs.setdefault("linewidth", 1.5)

    for i, roc in enumerate(rocs):
        color = colors[i % len(colors)]

        label = roc.name if roc.name else f"Model {i+1}"
        if show_auc:
            label += f" (AUC = {roc.auc:.3f})"

        # Core ROC Line
        (line,) = ax.plot(
            roc.fpr, roc.tpr, label=label, color=color, zorder=10 + i, **kwargs
        )

        # Shade Area Under Curve
        if shade_auc:
            if roc.partial_auc_range:
                min_r, max_r = roc.partial_auc_range
                if roc.partial_auc_focus == "specificity":
                    mask = (roc.fpr >= 1 - max_r) & (roc.fpr <= 1 - min_r)
                    ax.fill_between(
                        roc.fpr[mask],
                        0,
                        roc.tpr[mask],
                        color=color,
                        alpha=shade_alpha,
                        zorder=2 + i,
                    )
                else:
                    mask = (roc.tpr >= min_r) & (roc.tpr <= max_r)
                    ax.fill_between(
                        roc.fpr[mask],
                        0,
                        roc.tpr[mask],
                        color=color,
                        alpha=shade_alpha,
                        zorder=2 + i,
                    )
            else:
                ax.fill_between(
                    roc.fpr, 0, roc.tpr, color=color, alpha=shade_alpha, zorder=2 + i
                )

        # Shade Confidence Intervals
        if plot_ci:
            grid = np.linspace(0, 1, 100)
            if ci_type == "sensitivity":
                lower, upper = ci_sensitivity(roc, 1 - grid, n_boot=2000)
                ax.fill_between(
                    grid,
                    lower,
                    upper,
                    color=color,
                    alpha=ci_alpha,
                    linewidth=0,
                    zorder=5 + i,
                )
            elif ci_type == "specificity":
                lower, upper = ci_specificity(roc, grid, n_boot=2000)
                ax.fill_betweenx(
                    grid,
                    1 - upper,
                    1 - lower,
                    color=color,
                    alpha=ci_alpha,
                    linewidth=0,
                    zorder=5 + i,
                )

        # Highlight Optimal Operating Point
        if annotate_best:
            best_coords = roc.get_coords(
                x="best", best_method=best_method, ret=["specificity", "sensitivity"]
            )
            best_fpr = 1 - best_coords["specificity"]
            best_tpr = best_coords["sensitivity"]
            ax.scatter(
                best_fpr,
                best_tpr,
                color=color,
                s=40,
                zorder=15 + i,
                edgecolors="white",
                linewidths=1,
            )
            ax.annotate(
                f"({best_fpr:.2f}, {best_tpr:.2f})",
                (best_fpr, best_tpr),
                textcoords="offset points",
                xytext=(8, -8),
                fontsize=9,
                color=color,
            )

    ax.legend(loc="lower right")

    return ax
