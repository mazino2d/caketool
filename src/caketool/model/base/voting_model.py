import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class VotingModel(ClassifierMixin, BaseEstimator):
    """Generic ensemble that averages predictions from any fitted estimators.

    Unlike ``BaseEnsemble``, this class is not tied to ``BaseBoostTree`` and
    accepts any sklearn-compatible estimators. Estimators must already be
    fitted — ``fit()`` is a no-op.

    Parameters
    ----------
    estimators : list[BaseEstimator]
        Pre-fitted sklearn-compatible estimators.

    Examples
    --------
    >>> model = VotingModel([lgbm_model, xgb_model, catboost_model])
    >>> proba = model.predict_proba(X_test)[:, 1]
    """

    def __init__(self, estimators: list[BaseEstimator]):
        self.estimators = estimators

    def fit(self, X=None, y=None) -> "VotingModel":
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

    @property
    def classes_(self) -> np.ndarray:
        """Class labels, taken from the first estimator."""
        return self.estimators[0].classes_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the class with the highest averaged probability.

        Falls back to averaging ``predict`` outputs when the estimators do not
        expose ``predict_proba`` and ``classes_`` (e.g. regressors).

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        first = self.estimators[0]
        if hasattr(first, "predict_proba") and hasattr(first, "classes_"):
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average probability predictions across all estimators.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
