"""Model monitoring utilities for production deployments.

Provides:

- ``ModelMonitor`` – BigQuery-backed monitor for storing and comparing
  feature and score distributions across dataset versions.  Supports
  normalization, percentile-based binning, and distribution storage.
- ``AdversarialModel`` – Drift detector that trains a binary classifier
  to distinguish between two datasets.  A high ROC-AUC indicates
  significant distributional shift between the reference and the
  production population.
"""

from .adversarial_test import AdversarialModel as AdversarialModel
from .model_monitor import ModelMonitor as ModelMonitor
