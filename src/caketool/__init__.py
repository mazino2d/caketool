"""caketool — Machine Learning toolbox for credit risk modelling.

Provides end-to-end utilities for building, evaluating, and monitoring
credit scoring models, including:

- **eda** – Exploratory data analysis (profiles, univariate/bivariate plots,
  correlation matrices).
- **feature** – Window-based feature engineering across multiple backends
  (pandas, polars, Spark, BigQuery BigFrames).
- **model** – XGBoost-based binary and multiclass classifiers with a built-in
  preprocessing pipeline, ensemble utilities, and feature selection.
- **metric** – Association metrics (Pearson, Spearman, Eta, Cramér's V),
  Gini coefficient, and Population Stability Index (PSI).
- **calibration** – Post-hoc score calibration via normal distribution
  transformation (stateless function or fit/transform estimator).
- **explainability** – SHAP-based global and local model explanations.
- **experiment** – Unified experiment tracking interface for Vertex AI,
  MLflow, and Weights & Biases.
- **monitor** – Drift detection (adversarial classification).
- **report** – Risk score segmentation and reporting for credit decisions.
- **utils** – Dynamic imports and optional-dependency checks.

Quick start
-----------
>>> import caketool.eda as eda
>>> import pandas as pd
>>> df = pd.read_csv("data.csv")
>>> eda.profile(df)
"""

try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("caketool")
    except PackageNotFoundError:
        __version__ = "unknown"
