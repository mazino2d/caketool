"""Evaluation metrics for binary classification and data distribution monitoring.

Includes:

- ``association`` – pairwise association metrics (Pearson, Spearman, Eta,
  Cramér's V) for numeric and categorical variables.
- ``gini`` – Gini coefficient derived from ROC-AUC.
- ``psi`` / ``psi_from_distribution`` – Population Stability Index for
  monitoring score or feature distribution drift.

All scikit-learn metrics are also re-exported via ``from sklearn.metrics import *``.
"""

from sklearn.metrics import *  # noqa: F403

from .association_metric import association as association
from .classification_metric import gini as gini
from .stability_metric import psi as psi
from .stability_metric import psi_from_distribution as psi_from_distribution
