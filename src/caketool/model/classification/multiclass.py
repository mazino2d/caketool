from __future__ import annotations

import xgboost as xgb
from sklearn.base import ClassifierMixin

from caketool.model.base.boost_tree import BaseBoostTree
from caketool.model.config import ModelConfig


class MulticlassBoostTree(ClassifierMixin, BaseBoostTree):
    """XGBoost multiclass classifier with an integrated preprocessing pipeline.

    Extends ``BaseBoostTree`` with ``objective="multi:softprob"``.
    ``predict_proba`` returns a ``(n_samples, n_classes)`` array; ``predict``
    returns the argmax class index.

    Parameters
    ----------
    config : ModelConfig or None
        Model and preprocessing configuration. Uses ``ModelConfig()`` defaults
        when ``None``.
    num_class : int or None
        Number of target classes. When ``None``, inferred from ``y`` during
        ``fit`` via ``len(np.unique(y))``.

    Examples
    --------
    >>> model = MulticlassBoostTree()
    >>> model.fit(X_train, y_train)
    >>> proba = model.predict_proba(X_test)   # shape (n_samples, n_classes)
    """

    _xgb_class = xgb.XGBClassifier
    _default_objective = "multi:softprob"

    def __init__(self, config: ModelConfig | None = None, num_class: int | None = None) -> None:
        super().__init__(config=config)
        self.num_class = num_class

    def _build_model(self, cfg: ModelConfig):
        model = xgb.XGBClassifier(
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
            eval_metric="mlogloss",
            nthread=4,
        )
        # num_class is required for multi:softprob; when None, XGBoost infers it from data
        if self.num_class is not None:
            model.set_params(num_class=self.num_class)
        return model
