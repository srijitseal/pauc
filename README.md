# pAUC

[![PyPI](https://img.shields.io/pypi/v/pauc.svg)](https://pypi.org/project/pauc/)
[![Python Tests](https://github.com/srijitseal/pauc/actions/workflows/ci.yml/badge.svg)](https://github.com/srijitseal/pauc/actions/workflows/ci.yml)

A simple Python package to calculate ROC AUC confidence intervals using DeLong’s method.

## Installation

pip install pauc

## Usage

from pauc import roc_auc_ci_score

auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)
print(f'AUC: {auc}, 95% CI: ({lb}, {ub})')
