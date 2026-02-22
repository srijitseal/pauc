import pytest
import numpy as np
import pandas as pd
from pauc import (
    ROC,
    MultiClassROC,
    smooth,
    compare,
    ci_auc,
    ci_sensitivity,
    ci_specificity,
)
from pauc.stats import (
    venkatraman_test,
    test_operating_point as stat_test_operating_point,
)
from pauc.power import power_roc, sample_size_roc

try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    proc = importr("pROC")
    HAS_R = True
except ImportError:
    HAS_R = False

TOL = 1e-3


@pytest.fixture(scope="module")
def ames_data():
    """Loads the dataset once for all tests."""
    return pd.read_csv("./proc_validation_data/ames_comparison.csv")


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
@pytest.mark.parametrize(
    "model_col", ["pred_Minimol", "pred_Chemprop", "pred_MapLight"]
)
@pytest.mark.parametrize("paired", [True, False])
def test_delong_comparisons(ames_data, model_col, paired):
    """Exhaustively tests DeLong Z-scores and P-values across all models and pairing assumptions."""
    y_true = ames_data["y_true"].values
    pred_base = ames_data["pred_Baseline_ECFP4"].values
    pred_mod = ames_data[model_col].values

    # Python pAUC
    roc_base = ROC(y_true, pred_base, direction="<")
    roc_mod = ROC(y_true, pred_mod, direction="<")
    py_comp = compare(roc_mod, roc_base, method="delong", paired=paired)

    # R pROC
    r_y = robjects.FloatVector(y_true.tolist())
    r_base = robjects.FloatVector(pred_base.tolist())
    r_mod = robjects.FloatVector(pred_mod.tolist())
    r_roc_base = proc.roc(
        r_y, r_base, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_roc_mod = proc.roc(
        r_y, r_mod, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_comp = proc.roc_test(r_roc_mod, r_roc_base, method="delong", paired=paired)

    # Assertions
    assert np.isclose(
        roc_mod.auc, r_roc_mod.rx2("auc")[0], atol=TOL
    ), f"{model_col} AUC mismatch"
    assert np.isclose(
        py_comp.stat, r_comp.rx2("statistic")[0], atol=TOL
    ), f"{model_col} Z-score mismatch (paired={paired})"
    assert np.isclose(
        py_comp.p_value, r_comp.rx2("p.value")[0], atol=TOL
    ), f"{model_col} p-value mismatch (paired={paired})"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_confidence_intervals_delong(ames_data):
    """Tests calculation of DeLong Confidence Intervals."""
    y_true = ames_data["y_true"].values
    pred_mod = ames_data["pred_Minimol"].values

    # Python CI
    roc_mod = ROC(y_true, pred_mod, direction="<")
    py_ci = ci_auc(roc_mod, method="delong")

    # R CI
    r_y = robjects.FloatVector(y_true.tolist())
    r_mod = robjects.FloatVector(pred_mod.tolist())
    r_roc_mod = proc.roc(
        r_y, r_mod, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_ci = proc.ci_auc(r_roc_mod, method="delong")

    # R returns CI as a vector of (lower, median, upper)
    assert np.isclose(py_ci[0], r_ci[0], atol=TOL), "Lower CI mismatch"
    assert np.isclose(py_ci[1], r_ci[2], atol=TOL), "Upper CI mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
@pytest.mark.parametrize("focus", ["specificity", "sensitivity"])
def test_partial_auc(ames_data, focus):
    """Tests Partial AUC boundaries for both sensitivity and specificity focus."""
    y_true = ames_data["y_true"].values
    pred_mod = ames_data["pred_Minimol"].values

    # Setting range to 0.8 - 1.0 (Top 20% of the curve)
    py_roc = ROC(
        y_true, pred_mod, direction="<", partial_auc=(0.8, 1.0), partial_auc_focus=focus
    )

    r_y = robjects.FloatVector(y_true.tolist())
    r_mod = robjects.FloatVector(pred_mod.tolist())

    # Pack rpy2 arguments dynamically to bypass `_` to `.` resolution bugs
    kwargs = {
        "partial.auc": robjects.FloatVector([1.0, 0.8]),
        "partial.auc.focus": focus,
    }

    r_roc = proc.roc(
        r_y,
        r_mod,
        direction="<",
        levels=robjects.IntVector([0, 1]),
        quiet=True,
        **kwargs,
    )

    assert np.isclose(
        py_roc.auc, r_roc.rx2("auc")[0], atol=TOL
    ), f"pAUC mismatch for {focus} focus. Py={py_roc.auc}, R={r_roc.rx2('auc')[0]}"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_multiclass_auc():
    """Tests Hand and Till averaging for multiclass ROC."""
    np.random.seed(42)
    # Mock 3-class data
    y_true = np.random.choice([0, 1, 2], size=150)
    # Mock prediction matrix (150, 3)
    y_score_matrix = np.random.rand(150, 3)
    # For R's multiclass.roc, we can pass a single predictor vector (if ordering matters)
    # or test the algorithm logic directly. We'll use a single continuous predictor for parity.
    pred_single = y_score_matrix[:, 1]

    py_roc = MultiClassROC(
        y_true, np.column_stack([pred_single] * 3)
    )  # Python expects matrix, but computes pairs
    # Wait, R's multiclass takes a single vector. To compare apples to apples:
    # We rebuild Python's multiclass using the single vector broadcasted.
    py_roc_single = MultiClassROC(y_true, np.tile(pred_single, (3, 1)).T)

    r_y = robjects.IntVector(y_true.tolist())
    r_pred = robjects.FloatVector(pred_single.tolist())
    r_multi = proc.multiclass_roc(r_y, r_pred, quiet=True)

    assert np.isclose(
        py_roc_single.auc, r_multi.rx2("auc")[0], atol=TOL
    ), "Multiclass AUC mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_get_coords_best(ames_data):
    y_true = ames_data["y_true"].values
    pred = ames_data["pred_Minimol"].values

    py_roc = ROC(y_true, pred, direction="<")
    py_coords = py_roc.get_coords(
        x="best", ret=["threshold", "specificity", "sensitivity"]
    )

    r_y = robjects.FloatVector(y_true.tolist())
    r_pred = robjects.FloatVector(pred.tolist())
    r_roc = proc.roc(
        r_y, r_pred, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )

    r_coords = proc.coords(
        r_roc,
        "best",
        ret=robjects.StrVector(["threshold", "specificity", "sensitivity"]),
        transpose=False,
    )

    # R can return multiple thresholds if they are exactly tied for max Youden J.
    # Check if Python's chosen threshold exists within R's list of valid tied thresholds.
    r_thresh = np.array(r_coords.rx2("threshold"))
    assert any(
        np.isclose(py_coords["threshold"], t, atol=TOL) for t in r_thresh
    ), "Threshold tie mismatch"

    r_spec = np.array(r_coords.rx2("specificity"))
    assert any(
        np.isclose(py_coords["specificity"], s, atol=TOL) for s in r_spec
    ), "Specificity tie mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
@pytest.mark.parametrize("method", ["binormal", "density"])
def test_smoothing(ames_data, method):
    y_true = ames_data["y_true"].values
    pred = ames_data["pred_Minimol"].values

    py_roc = ROC(y_true, pred, direction="<")
    py_smooth = smooth(py_roc, method=method)

    r_y = robjects.FloatVector(y_true.tolist())
    r_pred = robjects.FloatVector(pred.tolist())
    r_roc = proc.roc(
        r_y, r_pred, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_smooth = proc.smooth(r_roc, method=method)

    # Allow slight variance for density (KDE internal algorithms differ between languages)
    tol = 0.01 if method == "density" else TOL
    assert np.isclose(
        py_smooth.auc, r_smooth.rx2("auc")[0], atol=tol
    ), f"{method} smoothing AUC mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_power_and_sample_size():
    auc_val, power_target, kappa_val = 0.80, 0.80, 2.0

    py_cases, py_controls = sample_size_roc(
        auc_val, power=power_target, kappa=kappa_val
    )
    r_ss = proc.power_roc_test(
        auc=auc_val, power=power_target, kappa=kappa_val, sig_level=0.05
    )

    # Check continuous evaluation parity with fractional allowance for root finding diffs
    assert py_cases == pytest.approx(
        r_ss.rx2("ncases")[0], abs=0.5
    ), "Sample size (cases) mismatch"
    assert py_controls == pytest.approx(
        r_ss.rx2("ncontrols")[0], abs=1.0
    ), "Sample size (controls) mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_stochastic_methods(ames_data):
    df = ames_data.sample(300, random_state=42)
    y_true, pred1, pred2 = (
        df["y_true"].values,
        df["pred_Minimol"].values,
        df["pred_Baseline_ECFP4"].values,
    )

    py_roc1, py_roc2 = ROC(y_true, pred1, direction="<"), ROC(
        y_true, pred2, direction="<"
    )

    py_boot = compare(py_roc1, py_roc2, method="bootstrap", n_boot=500, paired=True)
    py_venk = venkatraman_test(py_roc1, py_roc2, n_perm=500, paired=True)

    r_y = robjects.FloatVector(y_true.tolist())
    r_p1, r_p2 = robjects.FloatVector(pred1.tolist()), robjects.FloatVector(
        pred2.tolist()
    )
    r_roc1 = proc.roc(
        r_y, r_p1, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_roc2 = proc.roc(
        r_y, r_p2, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )

    r_boot = proc.roc_test(
        r_roc1, r_roc2, method="bootstrap", boot_n=500, paired=True, progress="none"
    )
    r_venk = proc.roc_test(
        r_roc1, r_roc2, method="venkatraman", boot_n=500, paired=True, progress="none"
    )

    # Relaxed tolerance for 500 n_boot iterations across different RNG engines
    assert np.isclose(
        py_boot.stat, r_boot.rx2("statistic")[0], atol=1e-1
    ), "Bootstrap Z mismatch"
    assert np.isclose(
        py_venk.p_value, r_venk.rx2("p.value")[0], atol=1e-1
    ), "Venkatraman p-value mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
@pytest.mark.parametrize("focus", ["specificity", "sensitivity"])
def test_standardized_pauc(ames_data, focus):
    y_true = ames_data["y_true"].values
    pred = ames_data["pred_Minimol"].values

    py_roc = ROC(
        y_true,
        pred,
        direction="<",
        partial_auc=(0.8, 1.0),
        partial_auc_focus=focus,
        standardize_pauc=True,
    )

    r_y = robjects.FloatVector(y_true.tolist())
    r_pred = robjects.FloatVector(pred.tolist())
    kwargs = {
        "partial.auc": robjects.FloatVector([1.0, 0.8]),
        "partial.auc.focus": focus,
        "partial.auc.correct": True,
    }
    r_roc = proc.roc(
        r_y,
        r_pred,
        direction="<",
        levels=robjects.IntVector([0, 1]),
        quiet=True,
        **kwargs,
    )

    # Standardization divides by (max_area - min_area), amplifying baseline interpolation diffs
    assert np.isclose(
        py_roc.auc, r_roc.rx2("auc")[0], atol=1e-2
    ), f"Standardized pAUC mismatch ({focus})"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_get_coords_topleft(ames_data):
    """Tests alternative 'closest top-left' method for optimal threshold."""
    y_true = ames_data["y_true"].values
    pred = ames_data["pred_Minimol"].values

    py_roc = ROC(y_true, pred, direction="<")
    py_coords = py_roc.get_coords(
        x="best", best_method="topleft", ret=["threshold", "specificity"]
    )

    r_y = robjects.FloatVector(y_true.tolist())
    r_pred = robjects.FloatVector(pred.tolist())
    r_roc = proc.roc(
        r_y, r_pred, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )

    r_coords = proc.coords(
        r_roc,
        "best",
        best_method="closest.topleft",
        ret=robjects.StrVector(["threshold", "specificity"]),
        transpose=False,
    )

    r_thresh = np.array(r_coords.rx2("threshold"))
    assert any(
        np.isclose(py_coords["threshold"], t, atol=TOL) for t in r_thresh
    ), "Top-left threshold mismatch"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_bootstrap_cis(ames_data):
    """Tests Bootstrap CIs for AUC, Sensitivity, and Specificity."""
    df = ames_data.sample(200, random_state=42)
    y_true, pred = df["y_true"].values, df["pred_Minimol"].values
    py_roc = ROC(y_true, pred, direction="<")

    # 1. AUC Bootstrap CI
    py_ci_auc = ci_auc(py_roc, method="bootstrap", n_boot=200)

    r_y = robjects.FloatVector(y_true.tolist())
    r_pred = robjects.FloatVector(pred.tolist())
    r_roc = proc.roc(
        r_y, r_pred, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_ci_auc = proc.ci_auc(r_roc, method="bootstrap", boot_n=200, progress="none")

    # Stochastic bounds check
    assert np.isclose(
        py_ci_auc[0], r_ci_auc[0], atol=1e-2
    ), "Boot CI AUC lower mismatch"

    # 2. Coordinate CIs (Execution checks)
    py_ci_sens = ci_sensitivity(py_roc, [0.90], n_boot=100)
    py_ci_spec = ci_specificity(py_roc, [0.90], n_boot=100)
    assert (
        len(py_ci_sens[0]) == 1 and len(py_ci_spec[0]) == 1
    ), "Coordinate CI shape error"


@pytest.mark.skipif(not HAS_R, reason="Requires R and pROC")
def test_test_operating_point(ames_data):
    df = ames_data.sample(200, random_state=42)
    y_true, pred1, pred2 = (
        df["y_true"].values,
        df["pred_Minimol"].values,
        df["pred_Baseline_ECFP4"].values,
    )

    roc1, roc2 = ROC(y_true, pred1, direction="<"), ROC(y_true, pred2, direction="<")

    # Python paired bootstrap
    py_test = stat_test_operating_point(
        roc1, roc2, point=0.90, point_type="specificity", n_boot=500, paired=True
    )

    r_y = robjects.FloatVector(y_true.tolist())
    r_p1, r_p2 = robjects.FloatVector(pred1.tolist()), robjects.FloatVector(
        pred2.tolist()
    )
    r_roc1 = proc.roc(
        r_y, r_p1, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )
    r_roc2 = proc.roc(
        r_y, r_p2, direction="<", levels=robjects.IntVector([0, 1]), quiet=True
    )

    # R paired bootstrap
    r_test = proc.roc_test(
        r_roc1,
        r_roc2,
        method="specificity",  # MUST USE "specificity" instead of "bootstrap" here!
        boot_n=500,
        boot_stratified=False,
        specificity=0.90,
        paired=True,
        progress="none",
    )

    assert np.isclose(
        py_test.p_value, r_test.rx2("p.value")[0], atol=1e-1
    ), "Operating point p-value mismatch"
