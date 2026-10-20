"""caketool — Machine learning toolkit for data scientists.

Provides end-to-end utilities for building, evaluating, explaining and
monitoring machine learning models, including:

- **eda** – Exploratory data analysis (profiles, univariate/bivariate plots,
  correlation matrices).
- **feature** – Window-based aggregate feature engineering on pandas, with
  per-column choice of aggregations.
- **model** – Feature selection (univariate, collinear) and out-of-fold
  training for any scikit-learn compatible estimator.
- **metric** – Association metrics (Pearson, Spearman, Eta, Cramér's V),
  Gini coefficient, and Population Stability Index (PSI).
- **calibration** – Post-hoc score calibration via normal distribution
  transformation (stateless function or fit/transform estimator).
- **explainability** – SHAP-based global and local model explanations.
- **experiment** – Unified experiment tracking interface for Vertex AI,
  MLflow, and Weights & Biases.
- **monitor** – Drift detection (adversarial classification).
- **report** – Risk score segmentation and band reporting.
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
