# pAUC

Simple Python package to calculate ROC AUC with confidence intervals using DeLong’s method.

## Installation

pip install pauc

## Usage

from pauc import roc_auc_ci_score

auc, (lb, ub) = roc_auc_ci_score(y_true, y_pred)
print(f'AUC: {auc}, 95% CI: ({lb}, {ub})')