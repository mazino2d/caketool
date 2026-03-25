# Caketool

A Python MLOps toolkit for common machine learning and data science workflows.
Provides feature engineering, model training, experiment tracking, model monitoring, calibration, and metrics — all designed for production credit-risk and ML pipelines.

---

## Installation

```bash
# Core (pandas, numpy, sklearn, xgboost)
pip install caketool

# With MLflow support
pip install "caketool[onprem]"

# With Google Cloud (Vertex AI, BigQuery)
pip install "caketool[gcp]"

# Everything
pip install "caketool[all]"
```

---

## Quick Start

### Feature Generation

```python
from caketool.feature import generate_features_by_window

result = generate_features_by_window(
    df,
    client_id_col="user_id",
    report_date_col="event_date",
    fs_event_timestamp="snapshot_date",
    numeric_cols=("amount", "balance"),
    string_cols=("category",),
    boolean_cols=("is_active",),
    lookback_days=(0, 7, 30),   # 0 = lifetime
    backend="pandas",           # "pandas" | "polars" | "spark" | "bigframes"
)
```

### Model Training

```python
from caketool.model import BoostTree

model = BoostTree()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
proba = model.predict_proba(X_test)[:, 1]
importance = model.get_feature_importance()
```

### Out-of-Fold Cross-Validation

```python
from caketool.model import BoostTree, EnsembleBoostTree

models, oof_preds, oof_labels = BoostTree.fit_oof(X_train, y_train, n_splits=5)
ensemble = EnsembleBoostTree(models)
proba = ensemble.predict_proba(X_test)[:, 1]
```

### Score Calibration

```python
from caketool.calibration import calibrate_score_to_normal

calibrated = calibrate_score_to_normal(raw_scores, standard=False)
```

### Metrics

```python
from caketool.metric import gini, psi

print(gini(y_true, y_pred))          # Gini coefficient
print(psi(expected, actual))          # Population Stability Index
```

### Risk Report

```python
from caketool.report import decribe_risk_score

report = decribe_risk_score(score_df, pred_col="score", label_col="label")
```

### Drift Detection

```python
from caketool.monitor import AdversarialModel

model = AdversarialModel()
model.fit(reference_df, current_df)
model.show()    # prints ROC AUC and top important features
```

### Experiment Tracking

```python
from caketool.experiment import create_tracker

# MLflow
with create_tracker("mlflow", experiment_name="my-exp", run_name="run-001") as tracker:
    tracker.log_params({"lr": 0.01, "depth": 6})
    tracker.log_metrics({"gini": 0.72})
    tracker.log_pickle(model, "model")

# Vertex AI
with create_tracker("vertex_ai", experiment_name="my-exp", run_name="run-001",
                    project="my-gcp-project", location="us-central1",
                    bucket_name="my-bucket") as tracker:
    tracker.log_params({"lr": 0.01})
```

---

## API Overview

| Module | Key exports | Description |
| ------ | ----------- | ----------- |
| `caketool.feature` | `generate_features_by_window` | Multi-backend aggregated feature engineering |
| `caketool.model` | `BoostTree`, `EnsembleBoostTree`, `VotingModel` | XGBoost training & ensemble |
| `caketool.model` | `FeatureEncoder`, `FeatureRemover`, `ColinearFeatureRemover`, `UnivariateFeatureRemover`, `InfinityHandler` | sklearn-compatible preprocessing transformers |
| `caketool.calibration` | `calibrate_score_to_normal` | Normal distribution score calibration |
| `caketool.metric` | `gini`, `psi`, `psi_from_distribution` | Classification and stability metrics |
| `caketool.report` | `decribe_risk_score` | Risk score band report |
| `caketool.monitor` | `AdversarialModel` | Dataset drift detection |
| `caketool.experiment` | `create_tracker`, `MLflowTracker`, `VertexAITracker` | Experiment tracking abstraction |

---

## Development

```bash
conda create -n caketool python=3.10
conda activate caketool
pip-compile pyproject.toml --all-extras
pip install -e ".[dev,all]"
pre-commit install
```

### Linting

Pre-commit hooks run ruff automatically on commit. To run manually:

```bash
ruff check src/ tests/ --fix  # Lint and auto-fix
ruff format src/ tests/        # Format code
pre-commit run --all-files     # Run all hooks
```

### Tests

```bash
pytest tests/ -v --tb=short
```

---

## Publishing

Version is automatically derived from git tags via `setuptools-scm`.

```bash
# Test on TestPyPI (RC/beta/alpha tags)
git tag v1.8.0-rc1
git push origin v1.8.0-rc1

# Publish to PyPI (stable tags)
git tag v1.8.0
git push origin v1.8.0
```

GitHub Actions builds and publishes automatically on tag push.

---

## Local Development

```bash
python -m pip install -e .
python -c "from caketool import __version__; print(__version__)"
```
