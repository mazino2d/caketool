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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Average predictions across all estimators.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
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
