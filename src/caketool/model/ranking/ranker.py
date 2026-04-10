from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from caketool.model.base.boost_tree import BaseBoostTree, _build_pipeline
from caketool.model.config import ModelConfig


class BoostRanker(BaseBoostTree):
    """XGBoost learning-to-rank model with an integrated preprocessing pipeline.

    Uses ``XGBRanker`` with ``objective="rank:ndcg"``. ``predict`` returns
    relevance scores (higher = more relevant); there is no ``predict_proba``.

    Ranking requires group information — pass ``qid`` (group IDs aligned with
    ``X``) to ``fit``.

    Parameters
    ----------
    config : ModelConfig or None
        Model and preprocessing configuration. Uses ``ModelConfig()`` defaults
        when ``None``.

    Examples
    --------
    >>> model = BoostRanker()
    >>> model.fit(X_train, y_train, qid=group_ids_train)
    >>> scores = model.predict(X_test)
    """

    _xgb_class = xgb.XGBRanker
    _default_objective = "rank:ndcg"

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config=config)

    def _build_model(self, cfg: ModelConfig):
        return xgb.XGBRanker(
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
            n_estimators=cfg.n_estimators,
            nthread=4,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        qid: pd.Series | np.ndarray | None = None,
        eval_set: list | None = None,
        verbose: bool = False,
    ) -> BoostRanker:
        """Fit the preprocessing pipeline and XGBRanker.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Relevance labels (integers, higher = more relevant).
        qid : pd.Series or np.ndarray, optional
            Group/query IDs, one per sample. Required by XGBRanker.
        eval_set : list of (X, y) tuples, optional
            Validation sets for monitoring.
        verbose : bool, optional
            Default ``False``.

        Returns
        -------
        self
        """
        from sklearn import config_context

        cfg = self._get_config()
        self.preprocess = _build_pipeline(cfg)
        self.model = self._build_model(cfg)

        self.input_features_ = list(X.columns)

        with config_context(transform_output="pandas"):
            X_pre = self.preprocess.fit_transform(X, y)
            if eval_set:
                eval_set = [(self.preprocess.transform(s[0]), s[1]) for s in eval_set]

        fit_kwargs: dict = {}
        if qid is not None:
            fit_kwargs["qid"] = qid

        self.model.fit(X_pre, y, eval_set=eval_set, verbose=verbose, **fit_kwargs)
        self._build_required_features()
        return self

    def predict_proba(self, X):
        raise NotImplementedError("BoostRanker does not support predict_proba. Use predict() for relevance scores.")
