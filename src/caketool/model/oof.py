from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import GroupKFold, RepeatedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.utils.validation import check_is_fitted


def _take(data, idx: np.ndarray):
    """Select rows of ``data`` by position (DataFrame, Series or array)."""
    if data is None:
        return None
    if isinstance(data, pd.DataFrame | pd.Series):
        return data.iloc[idx]
    return np.asarray(data)[idx]


def _is_classifier(estimator) -> bool:
    """``sklearn.base.is_classifier`` that also works for estimators without sklearn tags.

    Older xgboost releases with scikit-learn >= 1.6 raise ``AttributeError`` there.
    """
    try:
        return is_classifier(estimator)
    except AttributeError:
        return getattr(estimator, "_estimator_type", None) == "classifier"


def _make_splitter(cv, n_repeats: int, classifier: bool, groups, random_state: int | None):
    """Return a CV splitter for an int ``cv``, or ``cv`` itself."""
    if not isinstance(cv, int):
        return cv
    if groups is not None:
        if n_repeats != 1:
            raise ValueError("n_repeats > 1 is not supported with groups; pass a splitter as cv instead.")
        return GroupKFold(n_splits=cv)
    splitter = RepeatedStratifiedKFold if classifier else RepeatedKFold
    return splitter(n_splits=cv, n_repeats=n_repeats, random_state=random_state)


class OOFEnsemble(BaseEstimator):
    """Average the predictions of estimators trained on different folds.

    Parameters
    ----------
    estimators : list
        Fitted estimators sharing the same features and, for classifiers,
        the same ``classes_``.

    Examples
    --------
    >>> result = fit_oof(XGBClassifier(), X_train, y_train)
    >>> result.ensemble.predict_proba(X_test)[:, 1]
    """

    def __init__(self, estimators: list):
        self.estimators = estimators

    def __sklearn_is_fitted__(self) -> bool:
        return len(self.estimators) > 0

    @property
    def classes_(self) -> np.ndarray:
        return self.estimators[0].classes_

    def fit(self, X=None, y=None) -> "OOFEnsemble":
        """No-op: the estimators are already fitted."""
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Mean of the estimators' ``predict_proba``.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self)
        return np.mean([estimator.predict_proba(X) for estimator in self.estimators], axis=0)

    def predict(self, X) -> np.ndarray:
        """Predict with the averaged estimators.

        Classifiers return the class with the highest mean probability;
        regressors return the mean prediction.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        if _is_classifier(self.estimators[0]):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        return np.mean([estimator.predict(X) for estimator in self.estimators], axis=0)


@dataclass
class OOFResult:
    """Output of :func:`fit_oof`.

    Attributes
    ----------
    estimators : list
        One fitted estimator per fold, in split order.
    oof_predictions : np.ndarray
        Out-of-fold predictions aligned to the rows of ``X``. With repeated CV
        each row is the mean over the folds that held it out. Rows that no fold
        held out are ``NaN``.
    folds : list[tuple[np.ndarray, np.ndarray]]
        ``(train_idx, val_idx)`` positions for each fold.
    ensemble : OOFEnsemble
        The fold estimators wrapped for averaged prediction on new data.
    """

    estimators: list
    oof_predictions: np.ndarray
    folds: list[tuple[np.ndarray, np.ndarray]]
    ensemble: OOFEnsemble


def fit_oof(
    estimator: Any,
    X,
    y,
    cv=5,
    n_repeats: int = 1,
    groups=None,
    method: Literal["auto", "predict_proba", "predict", "decision_function"] = "auto",
    early_stopping_size: float | None = None,
    sample_weight=None,
    fit_params: dict | None = None,
    random_state: int | None = 42,
) -> OOFResult:
    """Train a clone of ``estimator`` per CV fold and collect out-of-fold predictions.

    Works with any scikit-learn compatible estimator (XGBoost, LightGBM,
    CatBoost, sklearn models, pipelines). Each fold's model only sees its
    training rows, so ``oof_predictions`` are an unbiased estimate of
    performance on unseen data and safe to use for stacking or calibration.

    Parameters
    ----------
    estimator : estimator
        Unfitted estimator; it is cloned for every fold.
    X : pd.DataFrame or array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target, matched to ``X`` by position.
    cv : int or CV splitter, optional
        Number of folds, or any object with a ``split(X, y, groups)`` method.
        An int uses ``RepeatedStratifiedKFold`` for classifiers,
        ``RepeatedKFold`` otherwise, and ``GroupKFold`` when ``groups`` is
        given. Default ``5``.
    n_repeats : int, optional
        Number of CV repetitions when ``cv`` is an int. Default ``1``.
    groups : array-like, optional
        Group labels; rows of a group never span train and validation.
    method : {"auto", "predict_proba", "predict", "decision_function"}, optional
        Prediction method for the held-out folds. ``"auto"`` uses
        ``predict_proba`` for classifiers and ``predict`` otherwise.
    early_stopping_size : float or None, optional
        When set, this fraction of each training fold is split off and passed
        as ``eval_set=[(X_es, y_es)]`` to ``fit``, for estimators that stop
        early on it (e.g. ``XGBClassifier(early_stopping_rounds=50)``). The
        held-out fold is never used for stopping. Only for estimators whose
        ``fit`` accepts ``eval_set``, not sklearn pipelines. Default ``None``.
    sample_weight : array-like of shape (n_samples,), optional
        Per-row weights, sliced per fold and passed to ``fit``.
    fit_params : dict, optional
        Extra keyword arguments passed to every ``fit`` call
        (e.g. ``{"verbose": False}``).
    random_state : int or None, optional
        Seed for the fold split (int ``cv``) and the early-stopping split.
        Default ``42``.

    Returns
    -------
    OOFResult
        Fold estimators, aligned OOF predictions, fold indices and the
        averaging ensemble.

    Raises
    ------
    ValueError
        If ``X`` and ``y`` have different lengths, or ``n_repeats > 1`` is
        combined with ``groups`` and an int ``cv``.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from caketool.metric import gini
    >>> result = fit_oof(
    ...     XGBClassifier(n_estimators=500, early_stopping_rounds=50),
    ...     X_train, y_train, cv=5, early_stopping_size=0.2, fit_params={"verbose": False},
    ... )
    >>> gini(y_train, result.oof_predictions[:, 1])
    >>> proba = result.ensemble.predict_proba(X_test)[:, 1]
    """
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"X and y have different lengths: {n_samples} != {len(y)}")

    classifier = _is_classifier(estimator)
    if method == "auto":
        method = "predict_proba" if classifier else "predict"
    splitter = _make_splitter(cv, n_repeats, classifier, groups, random_state)

    estimators: list = []
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    pred_sum = None
    pred_count = np.zeros(n_samples)

    for train_idx, val_idx in splitter.split(X, y, groups):
        X_train, y_train = _take(X, train_idx), _take(y, train_idx)
        kwargs = dict(fit_params or {})
        if sample_weight is not None:
            kwargs["sample_weight"] = _take(sample_weight, train_idx)

        if early_stopping_size is not None:
            positions = np.arange(len(train_idx))
            fit_pos, es_pos = train_test_split(
                positions,
                test_size=early_stopping_size,
                stratify=np.asarray(y_train) if classifier else None,
                random_state=random_state,
            )
            kwargs["eval_set"] = [(_take(X_train, es_pos), _take(y_train, es_pos))]
            if "sample_weight" in kwargs:
                kwargs["sample_weight"] = _take(kwargs["sample_weight"], fit_pos)
            X_train, y_train = _take(X_train, fit_pos), _take(y_train, fit_pos)

        model = clone(estimator).fit(X_train, y_train, **kwargs)
        pred = np.asarray(getattr(model, method)(_take(X, val_idx)), dtype=float)

        if pred_sum is None:
            pred_sum = np.zeros((n_samples, *pred.shape[1:]))
        pred_sum[val_idx] += pred
        pred_count[val_idx] += 1
        estimators.append(model)
        folds.append((train_idx, val_idx))

    if pred_sum is None:
        raise ValueError("The CV splitter produced no folds.")

    held_out = pred_count > 0
    oof_predictions = np.full_like(pred_sum, np.nan)
    oof_predictions[held_out] = pred_sum[held_out] / pred_count[held_out].reshape(-1, *([1] * (pred_sum.ndim - 1)))

    return OOFResult(
        estimators=estimators,
        oof_predictions=oof_predictions,
        folds=folds,
        ensemble=OOFEnsemble(estimators),
    )
