import numpy as np
from scipy.integrate import trapezoid


class ROC:
    def __init__(
        self,
        y_true,
        y_score,
        name=None,
        direction="auto",
        percent=False,
        partial_auc=None,
        partial_auc_focus="specificity",
        standardize_pauc=False,
        drop_intermediate=True,
    ):
        self.y_true = np.asarray(y_true)
        self.y_score = np.asarray(y_score)
        self.name = name
        self.percent = percent
        self.partial_auc_range = partial_auc
        self.partial_auc_focus = partial_auc_focus
        self.standardize_pauc = standardize_pauc

        unique_labels = np.unique(self.y_true)
        if len(unique_labels) != 2:
            raise ValueError("y_true must contain exactly two unique binary labels.")

        self.controls_label, self.cases_label = unique_labels[0], unique_labels[1]
        self.controls = self.y_score[self.y_true == self.controls_label]
        self.cases = self.y_score[self.y_true == self.cases_label]
        self.n_controls, self.n_cases = len(self.controls), len(self.cases)

        if direction == "auto":
            self.direction = (
                ">" if np.median(self.cases) < np.median(self.controls) else "<"
            )
        else:
            self.direction = direction

        if self.direction == ">":
            self.aligned_cases, self.aligned_controls = -self.cases, -self.controls
            self.aligned_scores = -self.y_score
        else:
            self.aligned_cases, self.aligned_controls = self.cases, self.controls
            self.aligned_scores = self.y_score

        self._calculate_roc_points(drop_intermediate)
        self.auc = self._calculate_auc()

    def _calculate_roc_points(self, drop_intermediate):
        desc_score_indices = np.argsort(self.aligned_scores)[::-1]
        self.sorted_scores = self.aligned_scores[desc_score_indices]
        self.sorted_y_true = self.y_true[desc_score_indices]

        distinct_indices = np.where(np.diff(self.sorted_scores))[0]
        threshold_indices = np.concatenate(
            [distinct_indices, [len(self.sorted_scores) - 1]]
        )

        # Extract original distinct scores
        original_unique = self.y_score[desc_score_indices][threshold_indices]

        # Calculate midpoints like R's pROC
        if len(original_unique) > 1:
            midpoints = (original_unique[:-1] + original_unique[1:]) / 2.0
        else:
            midpoints = np.array([])

        if self.direction == ">":
            self.thresholds = np.concatenate([[-np.inf], midpoints, [np.inf]])
        else:
            self.thresholds = np.concatenate([[np.inf], midpoints, [-np.inf]])

        tps = np.concatenate(
            [[0], np.cumsum(self.sorted_y_true == self.cases_label)[threshold_indices]]
        )
        fps = np.concatenate(
            [
                [0],
                np.cumsum(self.sorted_y_true == self.controls_label)[threshold_indices],
            ]
        )

        self.tpr = tps / self.n_cases
        self.fpr = fps / self.n_controls
        self.specificity = 1 - self.fpr

        if self.percent:
            self.tpr *= 100
            self.fpr *= 100
            self.specificity *= 100

    def _calculate_auc(self):
        if self.partial_auc_range is None:
            return trapezoid(self.tpr, self.fpr)

        min_r, max_r = self.partial_auc_range

        if self.partial_auc_focus == "specificity":
            low_fpr, high_fpr = 1 - max_r, 1 - min_r
            x_vals = np.sort(np.unique(np.concatenate([self.fpr, [low_fpr, high_fpr]])))
            x_vals = x_vals[(x_vals >= low_fpr) & (x_vals <= high_fpr)]
            y_vals = np.interp(x_vals, self.fpr, self.tpr)

            pauc = trapezoid(y_vals, x_vals)

            if self.standardize_pauc:
                max_area = high_fpr - low_fpr
                min_area = 0.5 * (high_fpr**2 - low_fpr**2)
                return 0.5 * (1 + (pauc - min_area) / (max_area - min_area))
            return pauc

        elif self.partial_auc_focus == "sensitivity":
            # Interpolate FPR boundaries based on constraints along the TPR axis
            x_vals = np.sort(np.unique(np.concatenate([self.tpr, [min_r, max_r]])))
            x_vals = x_vals[(x_vals >= min_r) & (x_vals <= max_r)]
            y_vals = np.interp(x_vals, self.tpr, self.fpr)

            # Integrate 1 - FPR (Specificity) horizontally over the TPR axis
            pauc = trapezoid(1 - y_vals, x_vals)

            if self.standardize_pauc:
                max_area = max_r - min_r
                min_area = (max_r - min_r) - 0.5 * (max_r**2 - min_r**2)
                return 0.5 * (1 + (pauc - min_area) / (max_area - min_area))
            return pauc

        return 0.0

    def get_coords(
        self,
        x="best",
        input="threshold",
        ret=["specificity", "sensitivity"],
        best_method="youden",
    ):
        if x == "best":
            if best_method == "youden":
                idx = np.argmax(self.tpr + (1 - self.fpr) - 1)
            elif best_method == "topleft":
                idx = np.argmin(self.fpr**2 + (1 - self.tpr) ** 2)
            x, input = self.thresholds[idx], "threshold"

        if input == "threshold":
            indices = (
                np.arange(len(self.thresholds))
                if x == "all"
                else [np.abs(self.thresholds - x).argmin()]
            )
        elif input == "specificity":
            target = x if not self.percent else x / 100.0
            indices = [np.abs((1 - self.fpr) - target).argmin()]
        elif input == "sensitivity":
            target = x if not self.percent else x / 100.0
            indices = [np.abs(self.tpr - target).argmin()]

        idx = np.array(indices)
        tp, fp = self.tpr[idx] * self.n_cases, self.fpr[idx] * self.n_controls
        tn, fn = self.n_controls - fp, self.n_cases - tp

        metrics_map = {
            "threshold": self.thresholds[idx],
            "specificity": 1 - self.fpr[idx],
            "sensitivity": self.tpr[idx],
            "accuracy": (tp + tn) / (self.n_cases + self.n_controls),
            "ppv": np.divide(
                tp, (tp + fp), out=np.zeros_like(tp), where=(tp + fp) != 0
            ),
            "npv": np.divide(
                tn, (tn + fn), out=np.zeros_like(tn), where=(tn + fn) != 0
            ),
            "fpr": self.fpr[idx],
            "tpr": self.tpr[idx],
        }
        out = {k: metrics_map[k] for k in ret if k in metrics_map}
        if x != "all" and not isinstance(x, (list, np.ndarray)):
            return {k: v[0] if isinstance(v, np.ndarray) else v for k, v in out.items()}
        return out

    def __repr__(self):
        name_str = f"'{self.name}' " if self.name else ""
        pauc_str = f" (pAUC={self.partial_auc_range})" if self.partial_auc_range else ""
        return (
            f"<ROC {name_str}AUC={self.auc:.4f}{pauc_str} "
            f"| Cases: {self.n_cases}, Controls: {self.n_controls}>"
        )


class MultiClassROC:
    def __init__(self, y_true, y_score_matrix):
        """Hand and Till (2001) Multiclass AUC"""
        self.y_true = np.asarray(y_true)
        self.y_score_matrix = np.asarray(
            y_score_matrix
        )  # shape: (n_samples, n_classes)
        self.classes = np.unique(self.y_true)
        self.n_classes = len(self.classes)

        self.aucs = {}
        total_auc = 0
        pairs = 0

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                c1, c2 = self.classes[i], self.classes[j]
                mask = (self.y_true == c1) | (self.y_true == c2)
                y_sub = self.y_true[mask]

                # c1 vs c2
                roc1 = ROC(y_sub == c1, self.y_score_matrix[mask, i])
                # c2 vs c1
                roc2 = ROC(y_sub == c2, self.y_score_matrix[mask, j])

                pair_auc = (roc1.auc + roc2.auc) / 2.0
                self.aucs[(c1, c2)] = pair_auc
                total_auc += pair_auc
                pairs += 1

        self.auc = total_auc / pairs

    def __repr__(self):
        return f"<MultiClassROC AUC={self.auc:.4f} | Classes: {self.n_classes}>"
