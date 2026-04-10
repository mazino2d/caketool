"""Tests for VotingModel."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from src.caketool.model.base.voting_model import VotingModel


class _ConstantEstimator(BaseEstimator):
    """Dummy estimator that always returns a fixed value."""

    def __init__(self, value: float):
        self.value = value

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.value), np.full(len(X), self.value)])


class TestVotingModel:
    def test_predict_averages_estimators(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        model = VotingModel([_ConstantEstimator(0.2), _ConstantEstimator(0.4)])
        np.testing.assert_allclose(model.predict(X), [0.3, 0.3, 0.3])

    def test_predict_proba_averages_estimators(self):
        X = pd.DataFrame({"a": [1, 2]})
        model = VotingModel([_ConstantEstimator(0.0), _ConstantEstimator(1.0)])
        result = model.predict_proba(X)

        np.testing.assert_allclose(result[:, 0], [0.5, 0.5])
        np.testing.assert_allclose(result[:, 1], [0.5, 0.5])

    def test_single_estimator_returns_its_predictions(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        model = VotingModel([_ConstantEstimator(0.7)])
        np.testing.assert_allclose(model.predict(X), [0.7, 0.7, 0.7])

    def test_fit_returns_self(self):
        model = VotingModel([_ConstantEstimator(0.5)])
        assert model.fit(pd.DataFrame({"a": [1, 2]})) is model
