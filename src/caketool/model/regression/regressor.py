from __future__ import annotations

import xgboost as xgb
from sklearn.base import RegressorMixin

from caketool.model.base.boost_tree import BaseBoostTree
from caketool.model.config import ModelConfig


class BoostRegressor(RegressorMixin, BaseBoostTree):
    """XGBoost regressor with an integrated preprocessing pipeline.

    Extends ``BaseBoostTree`` with ``objective="reg:squarederror"`` and
    ``RegressorMixin``. ``predict`` returns continuous values; there is no
    ``predict_proba``.

    Parameters
    ----------
    config : ModelConfig or None
        Model and preprocessing configuration. Uses ``ModelConfig()`` defaults
        when ``None``.

    Examples
    --------
    >>> model = BoostRegressor()
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    _xgb_class = xgb.XGBRegressor
    _default_objective = "reg:squarederror"

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config=config)

    def _build_model(self, cfg: ModelConfig):
        return xgb.XGBRegressor(
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
            eval_metric="rmse",
            nthread=4,
        )

    def predict_proba(self, X):
        raise NotImplementedError("BoostRegressor does not support predict_proba. Use predict() for continuous output.")
