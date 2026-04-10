from __future__ import annotations

import xgboost as xgb
from sklearn.base import ClassifierMixin

from caketool.model.base.boost_tree import BaseBoostTree
from caketool.model.config import ModelConfig


class BinaryBoostTree(ClassifierMixin, BaseBoostTree):
    """XGBoost binary classifier with an integrated preprocessing pipeline.

    Extends ``BaseBoostTree`` with ``objective="binary:logistic"`` and
    ``ClassifierMixin`` so it integrates correctly with all sklearn tooling
    (``cross_val_score``, ``GridSearchCV``, ``StratifiedKFold``, etc.).

    Parameters
    ----------
    config : ModelConfig or None
        Model and preprocessing configuration. Uses ``ModelConfig()`` defaults
        when ``None``.

    Examples
    --------
    >>> model = BinaryBoostTree()
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> proba = model.predict_proba(X_test)[:, 1]
    >>> print(model.get_required_input_features())
    >>> print(model.feature_schema_)
    """

    _xgb_class = xgb.XGBClassifier
    _default_objective = "binary:logistic"

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config=config)
