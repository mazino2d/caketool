# Caketool

A Python toolkit for data scientists working on machine learning and deep learning problems.
Provides EDA, feature engineering, model training, explainability, experiment tracking, drift detection, calibration, and metrics — all designed for production-ready ML pipelines.

Supports Python 3.10–3.12.

---

## Installation

```bash
# Core (pandas, numpy, scikit-learn, xgboost, shap, plotly)
pip install caketool

# Optional extras
pip install "caketool[onprem]"   # MLflow tracking
pip install "caketool[gcp]"      # Vertex AI, BigQuery, GCS
pip install "caketool[wandb]"    # Weights & Biases tracking

# Everything
pip install "caketool[all]"
```

---

## Quick Start

### EDA

```python
from caketool import eda

# Dataset overview
eda.profile(df)
eda.plot_correlations(df)

# Univariate
eda.plot_numeric_distribution(df["age"])
eda.plot_categorical_frequency(df["category"])

# Bivariate
eda.plot_scatter(df, x="income", y="spend", color_by="segment")
eda.plot_distribution_by_group(df, cat_col="segment", num_col="income", mode="box")
```

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
    lookback_days=(0, 7, 30),   # 0 = lifetime (all rows before the snapshot)
    # Optional: only these aggregations for these columns; the rest get every aggregation of their type
    aggs={"amount": ["sum", "max", "p50"], "balance": ["avg"], "category": ["nunique"]},
)
```

Only rows with `report_date < fs_event_timestamp` are used, for every window including lifetime.
Available aggregations: numeric `min max avg sum std cnt diff p25 p50 p75`, string/categorical `cnt nunique unique_ratio`, boolean `poscnt posratio`, date `firstdatediff lastdatediff daysbetween`. An empty list skips a column.

### Feature Selection

```python
from caketool.model import ColinearFeatureSelector, UnivariateFeatureSelector

univariate = UnivariateFeatureSelector(threshold=0.05).fit(X_train, y_train)  # F-test p-value
colinear = ColinearFeatureSelector(correlation_threshold=0.9).fit(univariate.transform(X_train), y_train)
X_selected = colinear.transform(univariate.transform(X_train))
```

Both are scikit-learn transformers (`dropped_cols_`, `selected_cols_`, `get_feature_names_out`), so they also work inside a `Pipeline`.

### Out-of-Fold Training

```python
from xgboost import XGBClassifier
from caketool.metric import gini
from caketool.model import fit_oof

result = fit_oof(
    XGBClassifier(n_estimators=500, early_stopping_rounds=50),
    X_train, y_train,
    cv=5,                       # int or any CV splitter; groups=... for GroupKFold
    early_stopping_size=0.2,    # early stopping on a split of each training fold
    fit_params={"verbose": False},
)
print(gini(y_train, result.oof_predictions[:, 1]))   # OOF predictions aligned to X_train rows
proba = result.ensemble.predict_proba(X_test)[:, 1]  # mean of the fold models
```

`fit_oof` works with any scikit-learn compatible estimator or pipeline. Each fold's model only sees its training rows, so the OOF predictions are unbiased and can feed stacking or calibration.

### Explainability

```python
from caketool.explainability import PermutationExplainer

explainer = PermutationExplainer(model=model)  # pass class_index=k for multiclass models
explainer.fit(X_test)

# Global feature importance
importance = explainer.get_feature_importance()

# Local explanation for a single sample
local = explainer.get_local_explanation(row_index=0)

# Visualize
explainer.show_summary()
explainer.show_waterfall(row_index=0)
explainer.show_dependence(feature="income")
```

### Score Calibration

```python
from caketool.calibration import NormalScoreCalibrator, calibrate_score_to_normal

# Stateless
calibrated = calibrate_score_to_normal(raw_scores)

# Standardised with statistics from a reference set (safe for per-row scoring)
calibrator = NormalScoreCalibrator().fit(train_scores)
calibrated = calibrator.transform(new_scores)
```

### Metrics

```python
from caketool.metric import gini, psi

print(gini(y_true, y_pred))          # Gini coefficient, in [-1, 1]
print(psi(expected, actual))          # Population Stability Index
```

`psi` uses open-ended outer bins, so values outside the reference range are counted, and missing values get their own bin.

### Risk Report

```python
from caketool.report import describe_risk_score

report = describe_risk_score(score_df, pred_col="score", label_col="label")
```

### Drift Detection

```python
from caketool.monitor import AdversarialModel

adv = AdversarialModel(random_state=0)
adv.fit(reference_df, current_df)
print(adv.auc_score)               # ~0.5 = no drift, close to 1 = strong drift
drift = adv.get_drift_features()   # features ranked by contribution to drift
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
| `caketool.eda` | `profile`, `plot_correlations`, `plot_scatter`, `plot_distribution_by_group`, `rank_associations` | Exploratory data analysis with Plotly |
| `caketool.feature` | `generate_features_by_window` | Windowed aggregate features (pandas), configurable per column |
| `caketool.model` | `UnivariateFeatureSelector`, `ColinearFeatureSelector`, `fit_oof`, `OOFEnsemble`, `OOFResult` | Feature selection and out-of-fold training |
| `caketool.explainability` | `PermutationExplainer` | SHAP-based model-agnostic explainability |
| `caketool.calibration` | `calibrate_score_to_normal`, `NormalScoreCalibrator` | Normal distribution score calibration |
| `caketool.metric` | `association`, `gini`, `psi`, `psi_from_distribution` | Association, classification and stability metrics |
| `caketool.report` | `describe_risk_score` | Risk score band report |
| `caketool.monitor` | `AdversarialModel` | Dataset drift detection |
| `caketool.experiment` | `create_tracker`, `MLflowTracker`, `VertexAITracker`, `WandbTracker` | Experiment tracking abstraction |

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

### Docs

```bash
pip install -e ".[docs]"
pdoc src/caketool   # Preview at http://localhost:8080
```

The API reference is generated from docstrings with pdoc and published to [GitHub Pages](https://mazino2d.github.io/caketool) on every push to `main`.

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
