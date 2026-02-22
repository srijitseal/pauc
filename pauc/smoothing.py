import numpy as np
import scipy.stats
from .roc_core import ROC
from scipy.integrate import trapezoid


class SmoothedROC(ROC):
    def __init__(self, original, fpr, tpr, auc=None):
        self.name = f"Smoothed {original.name}" if original.name else "Smoothed ROC"
        self.fpr = fpr
        self.tpr = tpr
        self.percent = original.percent
        self.n_cases = original.n_cases
        self.n_controls = original.n_controls
        self.direction = original.direction
        self.y_true = original.y_true
        self.y_score = original.y_score

        if auc is not None:
            self.auc = auc
        else:
            self.auc = self._calculate_auc()

    # Override calculation methods to do nothing as FPR/TPR are pre-set
    def _calculate_roc_points(self, drop):
        pass

    def _calculate_auc(self):
        # Simple trapezoid on the smoothed points
        # return np.trapz(self.tpr, self.fpr)
        return trapezoid(self.tpr, self.fpr)

    def __repr__(self):
        name_str = f"'{self.name}' " if self.name else ""
        return (
            f"<SmoothedROC {name_str}AUC={self.auc:.4f} "
            f"| Cases: {self.n_cases}, Controls: {self.n_controls}>"
        )


def _bw_nrd0(x):
    """
    bw = 0.9 * min(sd, IQR/1.34) * n^-0.2
    """
    if len(x) < 2:
        return 1.0

    sd_x = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25

    lo = min(sd_x, iqr / 1.34) if iqr > 0 else sd_x
    if lo == 0:
        lo = 1.0  # fallback if zero variance

    return 0.9 * lo * (len(x) ** -0.2)


def smooth(roc, method="binormal", n=2048, bandwidth=None):
    """
    Smooths an ROC curve.
    method: "binormal" (fit normal distributions) or "density" (KDE).
    """
    if method == "binormal":
        # Binormal assumption: transform scores to normal quantiles (probit)
        # Linear regression on probit(TPR) vs probit(FPR)
        # y = a + b*x

        # Avoid infinity in probit
        mask = (roc.fpr > 0) & (roc.fpr < 1) & (roc.tpr > 0) & (roc.tpr < 1)

        # Check sufficiency
        if np.sum(mask) < 2:
            # Not enough points, return original (or diagonal)
            # pROC behavior: warning and fallback.
            print(
                "Warning: ROC curve not smoothable (not enough points inside (0,1)). Returning original."
            )
            return roc

        x = scipy.stats.norm.ppf(roc.fpr[mask])
        y = scipy.stats.norm.ppf(roc.tpr[mask])

        # Regression
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)

        # Generate smoothed curve
        # Grid of FPRs
        grid_fpr = np.linspace(0, 1, n)
        # Prob(Y > thresh) = Phi( a + b * Phi^-1(FPR) )
        # TPR = Phi( intercept + slope * Phi^-1(FPR) )
        # Handle 0 and 1 boundaries
        grid_tpr = np.zeros_like(grid_fpr)

        # Inner points
        valid_idx = (grid_fpr > 0) & (grid_fpr < 1)
        z = scipy.stats.norm.ppf(grid_fpr[valid_idx])
        grid_tpr[valid_idx] = scipy.stats.norm.cdf(intercept + slope * z)
        grid_tpr[-1] = 1.0

        return SmoothedROC(roc, grid_fpr, grid_tpr)

    elif method == "density":
        # Gaussian Kernel Density Estimation
        # Bandwidth selection: nrd0 (default in R density())

        cases = roc.aligned_cases
        controls = roc.aligned_controls

        h_cases = _bw_nrd0(cases) if bandwidth is None else bandwidth
        h_controls = _bw_nrd0(controls) if bandwidth is None else bandwidth

        # Setup grid for scores
        # Range extended by 3*h to capture tails
        all_scores = np.concatenate([cases, controls])
        min_s, max_s = np.min(all_scores), np.max(all_scores)
        pad = 3 * max(h_cases, h_controls)
        grid_scores = np.linspace(min_s - pad, max_s + pad, n)

        # Calculate PDFs
        # PDF = (1/nh) * sum K((x-Xi)/h)
        def calc_pdf(data, h, grid):
            n_data = len(data)
            # Matrix broadcasting: (Grid, Data)
            # Optimization: Process in chunks if huge, but fine for typical ROC use
            z = (grid[:, None] - data[None, :]) / h
            k = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)  # Gaussian
            return np.sum(k, axis=1) / (n_data * h)

        pdf_cases = calc_pdf(cases, h_cases, grid_scores)
        pdf_controls = calc_pdf(controls, h_controls, grid_scores)

        # Integrate PDF to get CDF -> TPR/FPR
        # Using cumulative sum * dx
        dx = grid_scores[1] - grid_scores[0]

        # In ROC, TPR = P(Cases > x).
        # CDF is P(X <= x). So TPR = 1 - CDF.
        cdf_cases = np.cumsum(pdf_cases) * dx
        cdf_controls = np.cumsum(pdf_controls) * dx

        # Normalize to ensure 0-1 (numerical errors)
        cdf_cases /= cdf_cases[-1]
        cdf_controls /= cdf_controls[-1]

        # If direction ">", Score > Thresh means Case.
        # TPR = 1 - CDF(thresh)
        tpr_smooth = 1 - cdf_cases
        fpr_smooth = 1 - cdf_controls

        return SmoothedROC(roc, fpr_smooth[::-1], tpr_smooth[::-1])

    else:
        raise ValueError(f"Unknown smoothing method: {method}")
