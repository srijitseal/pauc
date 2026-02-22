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

## Contributing 

Contributions are welcome\! Whether it's bug reports, feature requests, or code contributions, please feel free to open an issue or pull request on our GitHub repository.

-----

## License 

This project is licensed under the MIT License. See the `LICENSE` file for details.
