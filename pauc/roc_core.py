import numpy as np
from scipy.integrate import trapezoid
from itertools import combinations


class ROC:
    """
    Represents a Receiver Operating Characteristic (ROC) curve.

    This class calculates the necessary components of a ROC curve (FPR, TPR, thresholds)
    from true binary labels and predicted scores. It also computes the Area Under the
    Curve (AUC).

    Attributes:
        y_true (np.ndarray): The ground truth binary labels.
        y_score (np.ndarray): The predicted scores or probabilities for the positive class.
        name (str): An optional name for the ROC curve, used for plotting legends.
        cases (np.ndarray): Scores for the positive class (label == 1).
        controls (np.ndarray): Scores for the negative class (label == 0).
        n_cases (int): The number of positive samples.
        n_controls (int): The number of negative samples.
        thresholds (np.ndarray): The thresholds used to compute the curve.
        fpr (np.ndarray): The false positive rates corresponding to each threshold.
        tpr (np.ndarray): The true positive rates corresponding to each threshold.
        auc (float): The Area Under the ROC Curve.
    """

    def __init__(self, y_true, y_score, name=None):
        """
        Initializes the ROC object and computes the curve.

        Args:
            y_true (list, np.ndarray): True binary labels (0 for negative, 1 for positive).
            y_score (list, np.ndarray): Target scores, can be probability estimates of the
                                       positive class, confidence values, or non-thresholded
                                       measure of decisions.
            name (str, optional): The name of the ROC curve. Defaults to None.
        """
        if not isinstance(y_true, np.ndarray):
            y_true = np.asarray(y_true)
        if not isinstance(y_score, np.ndarray):
            y_score = np.asarray(y_score)
        if y_true.ndim != 1 or y_score.ndim != 1:
            raise ValueError("y_true and y_score must be 1-dimensional.")
        if len(y_true) != len(y_score):
            raise ValueError("y_true and y_score must have the same length.")
        if len(np.unique(y_true)) != 2:
            raise ValueError("y_true must contain only two unique binary labels.")

        self.y_true = y_true
        self.y_score = y_score
        self.name = name

        # We assume labels are 0 and 1, or can be mapped to them.
        positive_label = np.max(y_true)
        self.cases = self.y_score[self.y_true == positive_label]
        self.controls = self.y_score[self.y_true != positive_label]

        if len(self.cases) == 0:
            raise ValueError("No positive samples found in y_true.")
        if len(self.controls) == 0:
            raise ValueError("No negative samples found in y_true.")

        self.n_cases = len(self.cases)
        self.n_controls = len(self.controls)

        self.thresholds, self.fpr, self.tpr = self._calculate_roc_points()
        self.auc = self._calculate_auc()

    def _calculate_roc_points(self):
        """Calculates the points of the ROC curve (FPR and TPR) for various thresholds."""
        distinct_scores = np.unique(self.y_score)
        thresholds = np.sort(distinct_scores)[::-1]

        tpr = np.zeros(len(thresholds) + 1)
        fpr = np.zeros(len(thresholds) + 1)
        tpr[0], fpr[0] = 0, 0  # Start at (0,0)

        for i, thresh in enumerate(thresholds):
            tp = np.sum(self.cases >= thresh)
            fp = np.sum(self.controls >= thresh)
            tpr[i + 1] = tp / self.n_cases
            fpr[i + 1] = fp / self.n_controls

        return thresholds, fpr, tpr

    def _calculate_auc(self, fpr=None, tpr=None):
        """Calculates the Area Under the Curve using the trapezoidal rule."""
        if fpr is None:
            fpr = self.fpr
        if tpr is None:
            tpr = self.tpr
        return trapezoid(tpr, fpr)

    def partial_auc(self, focus="specificity", bounds=(0.8, 1.0)):
        """Calculates the partial area under the curve."""
        if focus not in ["specificity", "sensitivity"]:
            raise ValueError("focus must be either 'specificity' or 'sensitivity'")

        min_bound, max_bound = sorted(bounds)

        if focus == "specificity":
            x_values = 1 - self.fpr
            y_values = self.tpr
        else:
            x_values = self.tpr
            y_values = 1 - self.fpr

        # Create a finer grid for interpolation to handle bounds precisely
        fine_x = np.linspace(x_values.min(), x_values.max(), 1000)
        fine_y = np.interp(fine_x, x_values, y_values)

        indices = np.where((fine_x >= min_bound) & (fine_x <= max_bound))
        bounded_x = fine_x[indices]
        bounded_y = fine_y[indices]

        if len(bounded_x) < 2:
            return 0.0

        if focus == "specificity":
            p_fpr, p_tpr = 1 - bounded_x, bounded_y
        else:
            p_fpr, p_tpr = bounded_y, bounded_x

        sort_order = np.argsort(p_fpr)
        return self._calculate_auc(fpr=p_fpr[sort_order], tpr=p_tpr[sort_order])

    def get_coords(self, x, input="threshold", best_method="youden"):
        """Returns coordinates of the ROC curve at specified points."""
        if x == "best":
            if best_method == "youden":
                best_idx = np.argmax(self.tpr - self.fpr)
            elif best_method == "closest_topleft":
                dist = np.sqrt((self.fpr**2) + ((1 - self.tpr) ** 2))
                best_idx = np.argmin(dist)
            else:
                raise ValueError("best_method must be 'youden' or 'closest_topleft'")

            threshold = self.thresholds[best_idx - 1] if best_idx > 0 else float("inf")
            return {
                "threshold": threshold,
                "specificity": 1 - self.fpr[best_idx],
                "sensitivity": self.tpr[best_idx],
            }

        if not isinstance(x, (int, float, np.number)):
            raise ValueError("x must be 'best' or a numeric value.")

        if input == "threshold":
            best_idx = np.argmin(np.abs(self.thresholds - x))
            idx = best_idx + 1
            return {
                "threshold": self.thresholds[best_idx],
                "specificity": 1 - self.fpr[idx],
                "sensitivity": self.tpr[idx],
            }
        elif input == "specificity":
            spec = 1 - self.fpr
            xp, fp = spec[::-1], self.tpr[::-1]  # make increasing
            return {"specificity": x, "sensitivity": np.interp(x, xp, fp)}
        elif input == "sensitivity":
            xp, fp = self.tpr, 1 - self.fpr
            sort_idx = np.argsort(xp)  # ensure increasing
            return {
                "sensitivity": x,
                "specificity": np.interp(x, xp[sort_idx], fp[sort_idx]),
            }
        else:
            raise ValueError(
                "input must be one of 'threshold', 'specificity', 'sensitivity'"
            )

    def __repr__(self):
        """Provides a user-friendly string representation of the ROC object."""
        header = f"ROC curve '{self.name}':" if self.name else "ROC curve:"
        return f"{header}\n - {self.n_cases} cases, {self.n_controls} controls\n - AUC: {self.auc:.3f}"


class SmoothedROC(ROC):
    """Represents a smoothed ROC curve."""

    def __init__(self, original_roc, smoothed_fpr, smoothed_tpr):
        self.y_true, self.y_score, self.name = (
            None,
            None,
            f"Smoothed {original_roc.name}" if original_roc.name else "Smoothed ROC",
        )
        self.cases, self.controls = None, None
        self.n_cases, self.n_controls = original_roc.n_cases, original_roc.n_controls
        self.thresholds = None
        self.fpr, self.tpr = smoothed_fpr, smoothed_tpr
        self.auc = self._calculate_auc()


class MultiClassROC:
    # ... (keep __init__, _calculate_multiclass_auc, and __repr__ the same) ...
    def __init__(self, y_true, y_score_probs):
        self.y_true = np.asarray(y_true)
        self.y_score_probs = np.asarray(y_score_probs)
        self.labels = np.unique(y_true)
        if len(self.labels) < 3:
            raise ValueError("MultiClassROC is for 3 or more classes.")
        if self.y_score_probs.shape[1] != len(self.labels):
            raise ValueError(
                "Number of columns in y_score_probs must match number of unique labels."
            )

        self.pairwise_rocs = {}
        self._calculate_pairwise_rocs()
        self.auc = self._calculate_multiclass_auc()

    def _calculate_pairwise_rocs(self):
        """Creates ROC objects for every pair of classes."""
        # Loop over the integer indices of the labels, not the labels themselves
        for i, j in combinations(range(len(self.labels)), 2):
            label1, label2 = self.labels[i], self.labels[j]

            indices = np.where((self.y_true == label1) | (self.y_true == label2))
            y_true_pair = self.y_true[indices]

            # Use the integer index 'j' to select the correct probability column
            y_score_pair = self.y_score_probs[indices[0], j]

            # Convert to binary
            y_true_binary = (y_true_pair == label2).astype(int)

            roc_name = f"'{label1}' vs '{label2}'"
            # The key for the dictionary can remain the label values
            self.pairwise_rocs[(label1, label2)] = ROC(
                y_true_binary, y_score_pair, name=roc_name
            )

    def _calculate_multiclass_auc(self):
        """Calculates the multiclass AUC using Hand and Till's formula."""
        auc_sum = sum(roc.auc for roc in self.pairwise_rocs.values())
        c = len(self.labels)
        return (2 / (c * (c - 1))) * auc_sum

    def __repr__(self):
        header = f"Multi-class ROC analysis ({len(self.labels)} classes)\n"
        pairwise_aucs = "\n".join(
            [
                f" - {roc.name}: AUC = {roc.auc:.3f}"
                for roc in self.pairwise_rocs.values()
            ]
        )
        return f"{header} - Average AUC (Hand & Till): {self.auc:.3f}\n\nPairwise AUCs:\n{pairwise_aucs}"

