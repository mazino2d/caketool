"""Model monitoring utilities for production deployments.

Provides:

- ``AdversarialModel`` – Drift detector that trains a binary classifier
  to distinguish between two datasets.  A high ROC-AUC indicates
  significant distributional shift between the reference and the
  production population.
"""

from .adversarial_test import AdversarialModel as AdversarialModel
