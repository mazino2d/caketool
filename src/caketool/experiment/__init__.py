"""Experiment tracking with a unified API across multiple backends.

Provides a consistent interface for logging hyperparameters, metrics, and
artifacts regardless of the underlying tracking platform.

Supported backends:

- **Vertex AI** (``VertexAITracker``) – Google Cloud Vertex AI experiments
  with GCS artifact storage.
- **MLflow** (``MLflowTracker``) – Self-hosted or managed MLflow tracking
  server with configurable artifact location.
- **Weights & Biases** (``WandbTracker``) – Wandb cloud tracking with
  project-based experiment organisation.

Use ``create_tracker`` as a factory function to avoid importing backend
classes directly.

Example
-------
>>> from caketool.experiment import create_tracker
>>> tracker = create_tracker(
...     backend="mlflow",
...     experiment_name="credit-model",
...     run_name="xgb-v1",
...     tracking_uri="http://localhost:5000",
... )
>>> with tracker:
...     tracker.log_params({"learning_rate": 0.05, "max_depth": 7})
...     tracker.log_metrics({"gini": 0.62})
"""

from .experiment_tracker import ExperimentTracker as ExperimentTracker
from .experiment_tracker import MLflowTracker as MLflowTracker
from .experiment_tracker import VertexAITracker as VertexAITracker
from .experiment_tracker import WandbTracker as WandbTracker
from .experiment_tracker import create_tracker as create_tracker
