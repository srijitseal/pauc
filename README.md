<p align="center">
  <img src="assets/logo.png" alt="pAUC Logo" width="200"/>
</p>

<h3 align="center">
pAUC: A simple Python package to calculate ROC AUC confidence intervals using DeLong’s method
</h3>



[](https://www.google.com/search?q=https://badge.fury.io/py/pauc)
[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=https://github.com/srijitseal/pauc)

`pAUC` is an intuitive Python library for creating, comparing, and visualizing Receiver Operating Characteristic (ROC) curves. It provides a clean, object-oriented interface designed for rigorous statistical analysis of binary and multi-class classifiers.

The library is built for researchers and data scientists who need reliable statistical tests and publication-quality plots with minimal effort. It implements several key methods for comparing models, and calculating confidence intervals.

-----

## Key Features 🔬

  * **ROC Curve Generation**: Easily create `ROC` objects from true labels and prediction scores.
  * **Statistical Comparison**: Compare two ROC curves using multiple methods:
      * **DeLong's test** for correlated or uncorrelated curves.
      * **Bootstrap-based tests** for flexible comparisons.
      * **Venkatraman's test** for a non-parametric alternative.
  * **Confidence Intervals**: Calculate CIs for AUC, partial AUC, and coordinates (sensitivity/specificity) using bootstrapping or DeLong's method.
  * **Partial AUC (pAUC)**: Analyze specific regions of the ROC curve, focusing on high specificity or high sensitivity.
  * **Multi-Class Analysis**: Native support for one-vs-one multi-class ROC analysis using Hand & Till's method.
  * **Curve Smoothing**: Smooth ROC curves using polynomial or binormal methods.
  * **Plotting**: A simple but powerful plotting function to visualize and annotate one or more curves.

-----

## Installation 💻

To install the package, clone the repository and use pip to install it in your local environment.

```bash
git clone https://github.com/srijitseal/pauc.git
cd pauc
pip install .
```

For development, you can install it in "editable" mode, which links the installation to your source files:

```bash
pip install -e .
```

`pAUC` requires the following packages:

  * `numpy`
  * `scipy`
  * `matplotlib`

-----

## Quick Start 

Let's walk through a typical workflow: creating two ROC curves, comparing them statistically, and visualizing the result.

### 1\. Import and Prepare Data

First, import the library and generate some sample data for two hypothetical models.

```python
import numpy as np
import pauc

# Generate ground truth labels and scores for two models
y_true = np.array([0] * 50 + [1] * 50)
np.random.seed(42)
y_score1 = np.concatenate([np.random.normal(0, 1, 50), np.random.normal(1.2, 1, 50)])
y_score2 = np.concatenate([np.random.normal(0.2, 1, 50), np.random.normal(1.5, 1, 50)])
```

### 2\. Create ROC Objects

Instantiate a `ROC` object for each model. The object automatically calculates the AUC and all the points on the curve.

```python
# Create ROC objects with names for plotting
roc1 = pauc.ROC(y_true, y_score1, name="Model 1")
roc2 = pauc.ROC(y_true, y_score2, name="Model 2")

# Printing an object gives a quick summary
print(roc1)
# Output:
# ROC curve 'Model 1':
#  - 50 cases, 50 controls
#  - AUC: 0.852
```

### 3\. Statistically Compare Models

Use the `pauc.compare()` function to determine if the difference between the two models' AUCs is statistically significant. The default method is DeLong's test for correlated curves.

```python
# Compare the two ROC curves
comparison_result = pauc.compare(roc1, roc2, method="delong")
print(comparison_result)
```

This will produce a detailed report of the statistical test:

```
DeLong's test for two correlated ROC curves

data:  Model 1 and Model 2
Z = -1.972, p-value = 0.049
alternative hypothesis: true difference in AUC is not equal to 0
95 percent confidence interval:
 -0.160 -0.000

sample estimates:
AUC of Model 1 AUC of Model 2
      0.852       0.932
```

### 4\. Plot the Results

Finally, use the `pauc.plot_roc()` function to visualize the curves. You can plot multiple `ROC` objects at once and add annotations like confidence interval bands and the optimal threshold.

```python
# Plot the two curves together with CI bands and best thresholds
pauc.plot_roc(
    [roc1, roc2],
    title="Comparison of Two Models",
    plot_ci=True,
    annotate_best=True
)
```

-----

## Advanced Usage 

`pAUC` supports a variety of advanced analyses.

### Partial AUC (pAUC)

To analyze the AUC in a specific specificity range (e.g., between 90% and 100%), use the `partial_auc()` method. You can also compute a confidence interval for this partial area.

```python
# Calculate pAUC for Model 1 where specificity is between 0.9 and 1.0
pauc_val = roc1.partial_auc(focus="specificity", bounds=(0.9, 1.0))
print(f"Partial AUC (0.9-1.0 spec): {pauc_val:.3f}")

# Get a bootstrap confidence interval for the pAUC
pauc_ci = pauc.ci_auc(roc1, method='bootstrap', bounds=(0.9, 1.0), focus="specificity")
print(f"95% CI for pAUC: ({pauc_ci[0]:.3f}, {pauc_ci[1]:.3f})")
```

### Multi-Class ROC

For classification problems with 3 or more classes, use `MultiClassROC`. It performs a one-vs-one analysis and calculates the average AUC according to Hand & Till's method.

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate multi-class data
X, y = make_classification(n_classes=3, n_samples=150, n_features=10, n_informative=5, random_state=42)
y_probs = LogisticRegression().fit(X, y).predict_proba(X)

# Perform multi-class analysis
multi_roc = pauc.MultiClassROC(y, y_probs)
print(multi_roc)
```

-----

## API Overview 

The main components of the `pAUC` library are:

  * `pauc.ROC`: The main class for creating and analyzing a single ROC curve.
  * `pauc.MultiClassROC`: A class for handling multi-class classification analysis.
  * `pauc.compare()`: The primary function for statistical comparison of two `ROC` objects.
  * `pauc.ci_auc()`: A function to compute confidence intervals for the AUC or pAUC.
  * `pauc.plot_roc()`: The main plotting function for visualizing `ROC` objects.
  * `pauc.smooth()`: A utility for smoothing a `ROC` curve.

-----

## Contributing 

Contributions are welcome\! Whether it's bug reports, feature requests, or code contributions, please feel free to open an issue or pull request on our GitHub repository.

-----

## License 

This project is licensed under the MIT License. See the `LICENSE` file for details.
