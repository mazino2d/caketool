from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sklearn.feature_selection import f_classif


@dataclass
class ModelConfig:
    """Configuration for all BoostTree-based models.

    Controls the preprocessing pipeline, feature selection, and XGBoost
    hyperparameters. Pass an instance to any ``BaseBoostTree`` subclass
    (``BinaryBoostTree``, ``MulticlassBoostTree``, ``BoostRegressor``,
    ``BoostRanker``).

    Parameters
    ----------
    encoder_name : str
        Fully qualified class name of the categorical encoder.
        Default ``"category_encoders.TargetEncoder"``.
    use_outlier_clipper : bool
        Whether to apply ``OutlierClipper`` as the first preprocessing step.
        Default ``False``.
    outlier_lower_quantile : float
        Lower percentile for ``OutlierClipper``. Default ``0.01``.
    outlier_upper_quantile : float
        Upper percentile for ``OutlierClipper``. Default ``0.99``.
    use_missing_imputer : bool
        Whether to apply ``MissingValueImputer`` after ``InfinityHandler``.
        Default ``False``.
    missing_strategy : {"median", "mean", "constant"}
        Imputation strategy for ``MissingValueImputer``. Default ``"median"``.
    missing_fill_value : float
        Fill value when ``missing_strategy="constant"``. Default ``-999``.
    correlation_threshold : float
        Absolute correlation threshold for ``ColinearFeatureRemover``.
        Default ``0.9``.
    univariate_threshold : float
        P-value threshold for ``UnivariateFeatureRemover``. Features with
        p-value above this are dropped. Default ``0.05``.
    random_state : int
        XGBoost random seed. Default ``8799``.
    max_depth : int
        Maximum tree depth. Default ``7``.
    eta : float
        Learning rate. Default ``0.05``.
    subsample : float
        Subsample ratio of the training data. Default ``0.65``.
    min_child_weight : int
        Minimum sum of instance weight in a child. Default ``16``.
    colsample_bytree : float
        Subsample ratio of columns per tree. Default ``0.5``.
    n_estimators : int
        Number of boosting rounds. Default ``300``.

    Examples
    --------
    >>> cfg = ModelConfig(max_depth=5, eta=0.03, use_outlier_clipper=True)
    >>> model = BinaryBoostTree(cfg)
    >>> model.fit(X_train, y_train)
    """

    # Encoder
    encoder_name: str = "category_encoders.TargetEncoder"
    # OutlierClipper (optional)
    use_outlier_clipper: bool = False
    outlier_lower_quantile: float = 0.01
    outlier_upper_quantile: float = 0.99
    # MissingValueImputer (optional)
    use_missing_imputer: bool = False
    missing_strategy: Literal["median", "mean", "constant"] = "median"
    missing_fill_value: float = -999.0
    # Feature selection
    correlation_threshold: float = 0.9
    univariate_threshold: float = 0.05
    univariate_score_func: callable = field(default=f_classif)
    # XGBoost
    random_state: int = 8799
    max_depth: int = 7
    eta: float = 0.05
    subsample: float = 0.65
    min_child_weight: int = 16
    colsample_bytree: float = 0.5
    n_estimators: int = 300
    scale_pos_weight: float = 1.0
