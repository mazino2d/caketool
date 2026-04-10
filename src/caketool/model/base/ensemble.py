from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from caketool.model.base.boost_tree import BaseBoostTree


class BaseEnsemble(BaseEstimator):
    """Ensemble of ``BaseBoostTree`` models that averages predictions.

    Each estimator handles its own feature slicing via
    ``required_input_features_``, so the ensemble automatically works even
    when individual models were trained on different feature subsets (e.g.
    OOF models).

    Parameters
    ----------
    estimators : list[BaseBoostTree]
        Fitted ``BaseBoostTree`` models to ensemble.

    Attributes
    ----------
    required_input_features_ : list[str]
        Union of all required features across estimators.

    Examples
    --------
    >>> models, _, _ = BinaryBoostTree.fit_oof(X_train, y_train)
    >>> ensemble = BaseEnsemble(models)
    >>> proba = ensemble.predict_proba(X_test)[:, 1]
    """

    def __init__(self, estimators: list[BaseBoostTree]) -> None:
        self.estimators = estimators

    @property
    def required_input_features_(self) -> list[str]:
        """Union of all required features across all estimators."""
        all_required: set[str] = set()
        for est in self.estimators:
            all_required.update(est.required_input_features_)
        return sorted(all_required)

    def fit(self, X=None, y=None) -> BaseEnsemble:
        """No-op fit — estimators are assumed to be pre-fitted.

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        self
        """
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average class predictions across all estimators.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        y_preds = [est.predict(X) for est in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average class probabilities across all estimators.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        y_preds = [est.predict_proba(X) for est in self.estimators]
        return np.mean(y_preds, axis=0)

    def get_feature_importance(self) -> pd.DataFrame:
        """Aggregate feature importances across all estimators.

        Each estimator's raw importance scores are summed (after outer-join on
        feature name), then percent columns are recomputed on the aggregate.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature_name``, ``gain``, ``cover``, ``total_gain``,
            ``total_cover``, ``weight``, ``gain_pct``, ``cover_pct``,
            ``total_gain_pct``, ``total_cover_pct``, ``weight_pct``.
        """
        score_types = ["gain", "cover", "total_gain", "total_cover", "weight"]
        agg: pd.DataFrame | None = None

        for est in self.estimators:
            sub = est.get_feature_importance()[["feature_name"] + score_types]
            if agg is None:
                agg = sub
            else:
                agg = agg.merge(sub, how="outer", on="feature_name", suffixes=("_x", "_y")).fillna(0)
                for col in score_types:
                    agg[col] = agg[f"{col}_x"] + agg[f"{col}_y"]
                agg = agg[["feature_name"] + score_types]

        for col in score_types:
            total = agg[col].sum()
            agg[f"{col}_pct"] = agg[col] / total if total > 0 else 0.0

        return agg

    def get_feature_names(self) -> list[str]:
        """Return the union of all feature names used across estimators."""
        return self.required_input_features_
