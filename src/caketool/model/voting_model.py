from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class VotingModel(BaseEstimator, RegressorMixin):
    """
    An ensemble learning model that combines multiple base estimators to make predictions.
    Implements both the BaseEstimator and RegressorMixin from scikit-learn.

    Parameters
    ----------
    estimators : List[BaseEstimator]
        A list of scikit-learn estimators that will be used to make predictions.
        Each estimator in the list should implement the `predict` method, and if `predict_proba` 
        is to be used, they should also implement the `predict_proba` method.
    """

    def __init__(self, estimators: List[BaseEstimator]):
        """
        Initialize the VotingModel with a list of base estimators.

        Parameters
        ----------
        estimators : List[BaseEstimator]
            A list of scikit-learn estimators.
        """
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        """
        Fit the model. This method does not perform any fitting and is included to 
        maintain compatibility with scikit-learn's interface.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def predict(self, X: pd.DataFrame):
        """
        Predict using the model by averaging the predictions of all base estimators.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        y_pred : np.ndarray
            The averaged predictions.
        """
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities using the model by averaging the probability 
        predictions of all base estimators. This method is applicable only if all 
        base estimators implement `predict_proba`.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        y_pred_proba : np.ndarray
            The averaged probability predictions.
        """
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
