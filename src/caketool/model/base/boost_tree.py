from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from caketool.model.config import ModelConfig
from caketool.model.preprocess import (
    ColinearFeatureRemover,
    FeatureEncoder,
    InfinityHandler,
    MissingValueImputer,
    OutlierClipper,
    UnivariateFeatureRemover,
)

if TYPE_CHECKING:
    pass

_DEFAULT_CONFIG = ModelConfig()


def _build_pipeline(cfg: ModelConfig) -> Pipeline:
    steps: list[tuple[str, BaseEstimator]] = []

    if cfg.use_outlier_clipper:
        steps.append(("outlier_clipper", OutlierClipper(cfg.outlier_lower_quantile, cfg.outlier_upper_quantile)))

    steps.append(("feature_encoder", FeatureEncoder(cfg.encoder_name)))
    steps.append(("infinity_handler", InfinityHandler()))

    if cfg.use_missing_imputer:
        steps.append(("missing_imputer", MissingValueImputer(cfg.missing_strategy, cfg.missing_fill_value)))

    steps.append(
        (
            "univariate_feature_remover",
            UnivariateFeatureRemover(score_func=cfg.univariate_score_func, threshold=cfg.univariate_threshold),
        )
    )
    steps.append(("colinear_feature_remover", ColinearFeatureRemover(cfg.correlation_threshold)))

    return Pipeline(steps)


class BaseBoostTree(BaseEstimator):
    """Abstract base for all XGBoost-based models with a preprocessing pipeline.

    Subclasses must define:

    - ``_xgb_class`` – the XGBoost estimator class (e.g. ``xgb.XGBClassifier``)
    - ``_default_objective`` – XGBoost ``objective`` string

    The preprocessing pipeline (configurable via ``ModelConfig``) applies in
    order:

    1. ``OutlierClipper`` (optional)
    2. ``FeatureEncoder``
    3. ``InfinityHandler``
    4. ``MissingValueImputer`` (optional)
    5. ``UnivariateFeatureRemover``
    6. ``ColinearFeatureRemover``

    After ``fit``, the following fitted attributes are available:

    - ``input_features_`` – original column names passed to ``fit``
    - ``required_input_features_`` – minimal columns needed at inference
    - ``feature_schema_`` – full lineage dict (input → required → model)
    - ``classes_`` – unique target values (classifiers only)

    Parameters
    ----------
    config : ModelConfig or None
        Model and preprocessing configuration. Uses ``ModelConfig()`` defaults
        when ``None``.
    """

    _xgb_class: type = xgb.XGBClassifier
    _default_objective: str = "binary:logistic"

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config

    def _get_config(self) -> ModelConfig:
        return copy.deepcopy(self.config) if self.config is not None else copy.deepcopy(_DEFAULT_CONFIG)

    def _build_model(self, cfg: ModelConfig):
        return self._xgb_class(
            objective=self._default_objective,
            random_state=cfg.random_state,
            booster="gbtree",
            tree_method="approx",
            grow_policy="lossguide",
            max_depth=cfg.max_depth,
            eta=cfg.eta,
            gamma=0.5,
            subsample=cfg.subsample,
            min_child_weight=cfg.min_child_weight,
            colsample_bytree=cfg.colsample_bytree,
            scale_pos_weight=cfg.scale_pos_weight,
            n_estimators=cfg.n_estimators,
            eval_metric="auc",
            nthread=4,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list | None = None,
        verbose: bool = False,
    ) -> BaseBoostTree:
        """Fit the preprocessing pipeline and XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Target labels.
        eval_set : list of (X, y) tuples, optional
            Validation sets passed to XGBoost for monitoring. Each set is
            preprocessed automatically.
        verbose : bool, optional
            Whether to print XGBoost training logs. Default ``False``.

        Returns
        -------
        self
        """
        cfg = self._get_config()
        self.preprocess = _build_pipeline(cfg)
        self.model = self._build_model(cfg)

        self.classes_ = np.unique(y)
        self.input_features_ = list(X.columns)

        with config_context(transform_output="pandas"):
            X_pre = self.preprocess.fit_transform(X, y)
            if eval_set:
                eval_set = [(self.preprocess.transform(s[0]), s[1]) for s in eval_set]

        self.model.fit(X_pre, y, eval_set=eval_set, verbose=verbose)
        self._build_required_features()
        return self

    def _build_required_features(self) -> None:
        univariate_dropped = set(self.preprocess.named_steps["univariate_feature_remover"].dropped_cols or [])
        colinear_dropped = set(self.preprocess.named_steps["colinear_feature_remover"].dropped_cols or [])
        all_dropped = univariate_dropped | colinear_dropped

        self.required_input_features_: list[str] = [f for f in self.input_features_ if f not in all_dropped]
        self.feature_schema_: dict[str, list[str]] = {
            "input": self.input_features_,
            "required": self.required_input_features_,
            "dropped_by_univariate": sorted(univariate_dropped),
            "dropped_by_colinear": sorted(colinear_dropped),
            "model_features": self.get_feature_names(),
        }

    def get_required_input_features(self) -> list[str]:
        """Return the minimal set of original features needed at inference.

        Use this to query only the necessary columns from your feature store
        instead of loading all raw features.

        Returns
        -------
        list[str]
        """
        return self.required_input_features_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using only the required input features.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all columns in ``required_input_features_``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        X_pre = self._preprocess_inference(X)
        return self.model.predict(X_pre)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities using only the required input features.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all columns in ``required_input_features_``.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        X_pre = self._preprocess_inference(X)
        return self.model.predict_proba(X_pre)

    def _preprocess_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.required_input_features_]
        with config_context(transform_output="pandas"):
            return self.preprocess.transform(X)

    def get_feature_names(self) -> list[str]:
        """Return the list of feature names used by the XGBoost booster."""
        return self.model.get_booster().feature_names

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances with raw and percent columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature_name``, ``gain``, ``cover``, ``total_gain``,
            ``total_cover``, ``weight``, ``gain_pct``, ``cover_pct``,
            ``total_gain_pct``, ``total_cover_pct``, ``weight_pct``.
        """
        score_types = ["gain", "cover", "total_gain", "total_cover", "weight"]
        feat_importance: pd.DataFrame | None = None

        for score_type in score_types:
            fi_dict = self.model.get_booster().get_score(importance_type=score_type)
            fi_tb = pd.DataFrame(list(fi_dict.items()), columns=["feature_name", score_type])
            feat_importance = fi_tb if feat_importance is None else feat_importance.merge(fi_tb, on="feature_name")

        for col in score_types:
            total = feat_importance[col].sum()
            feat_importance[f"{col}_pct"] = feat_importance[col] / total if total > 0 else 0.0

        return feat_importance

    @classmethod
    def fit_oof(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        config: ModelConfig | None = None,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: int = 42,
        early_stopping_rounds: int | None = None,
    ) -> tuple[list[BaseBoostTree], np.ndarray, np.ndarray]:
        """Fit models using out-of-fold (OOF) cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Target labels.
        config : ModelConfig, optional
            Model configuration. Uses defaults when ``None``.
        n_splits : int, optional
            Number of CV folds. Default ``5``.
        n_repeats : int, optional
            Number of times to repeat CV. Default ``1``.
        random_state : int, optional
            Random seed for fold splitting. Default ``42``.
        early_stopping_rounds : int or None, optional
            Activates early stopping. Training stops if the validation metric
            does not improve for this many rounds. Default ``None``.

        Returns
        -------
        models : list[BaseBoostTree]
            Fitted model for each fold.
        oof_predictions : np.ndarray
            Concatenated OOF probability predictions.
        oof_labels : np.ndarray
            Concatenated OOF true labels.
        """
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        oof_predictions: list[np.ndarray] = []
        oof_labels: list[np.ndarray] = []
        models: list[BaseBoostTree] = []

        fit_kwargs: dict = {}
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        for _, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, y))):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = cls(config)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kwargs)

            oof_predictions.append(model.predict_proba(X_val))
            oof_labels.append(y_val.to_numpy())
            models.append(model)

        return models, np.concatenate(oof_predictions), np.concatenate(oof_labels)
