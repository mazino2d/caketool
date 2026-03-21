"""Tests for model preprocessing and ensemble modules."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from src.caketool.model.feature_remover import ColinearFeatureRemover, FeatureRemover, UnivariateFeatureRemover
from src.caketool.model.infinity_handler import InfinityHandler
from src.caketool.model.voting_model import VotingModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": [5.0, 3.0, 1.0, 4.0, 2.0],
        }
    )


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


# ---------------------------------------------------------------------------
# FeatureRemover
# ---------------------------------------------------------------------------


class TestFeatureRemover:
    def test_removes_specified_columns(self, sample_df):
        remover = FeatureRemover(droped_cols=("a", "b"))
        result = remover.fit_transform(sample_df)

        assert "a" not in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_ignores_nonexistent_columns(self, sample_df):
        remover = FeatureRemover(droped_cols=("nonexistent",))
        result = remover.fit_transform(sample_df)

        assert list(result.columns) == list(sample_df.columns)

    def test_empty_droped_cols(self, sample_df):
        remover = FeatureRemover(droped_cols=())
        result = remover.fit_transform(sample_df)

        assert list(result.columns) == list(sample_df.columns)

    def test_remove_all_columns(self, sample_df):
        remover = FeatureRemover(droped_cols=("a", "b", "c"))
        result = remover.fit_transform(sample_df)

        assert len(result.columns) == 0

    def test_fit_returns_self(self, sample_df):
        remover = FeatureRemover(droped_cols=("a",))
        result = remover.fit(sample_df)

        assert result is remover

    def test_transform_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        remover = FeatureRemover(droped_cols=("a",))
        remover.fit_transform(sample_df)

        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# ColinearFeatureRemover
# ---------------------------------------------------------------------------


class TestColinearFeatureRemover:
    def test_removes_highly_correlated_feature(self):
        # b is almost perfectly correlated with a
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "b": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
                "c": [10.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 3.0, 7.0, 5.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        remover = ColinearFeatureRemover(correlation_threshold=0.9)
        remover.fit(df, y)
        result = remover.transform(df)

        # a and b are highly collinear — one should be dropped
        assert len(result.columns) < len(df.columns)
        assert "c" in result.columns

    def test_keeps_uncorrelated_features(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "x": np.random.rand(50),
                "y": np.random.rand(50),
                "z": np.random.rand(50),
            }
        )
        target = pd.Series(np.random.randint(0, 2, 50))

        remover = ColinearFeatureRemover(correlation_threshold=0.9)
        remover.fit(df, target)
        result = remover.transform(df)

        # Randomly generated features unlikely to exceed threshold
        assert len(result.columns) == 3

    def test_default_threshold_is_09(self):
        remover = ColinearFeatureRemover()
        assert remover.correlation_threshold == 0.9

    def test_fit_returns_self(self, sample_df):
        y = pd.Series([0, 1, 0, 1, 0])
        remover = ColinearFeatureRemover()
        result = remover.fit(sample_df, y)

        assert result is remover


# ---------------------------------------------------------------------------
# UnivariateFeatureRemover
# ---------------------------------------------------------------------------


class TestUnivariateFeatureRemover:
    def test_removes_irrelevant_features(self, classification_data):
        X, y = classification_data
        # Add a purely random (irrelevant) feature
        X = X.copy()
        np.random.seed(0)
        X["noise"] = np.random.rand(len(X))

        remover = UnivariateFeatureRemover(threshold=0.05)
        remover.fit(X, y)
        result = remover.transform(X)

        # The noise feature should be removed
        assert "noise" not in result.columns

    def test_keeps_relevant_features(self, classification_data):
        X, y = classification_data
        remover = UnivariateFeatureRemover(threshold=0.05)
        remover.fit(X, y)
        result = remover.transform(X)

        # At least some original informative features should remain
        assert len(result.columns) > 0

    def test_feature_importance_stored_after_fit(self, classification_data):
        X, y = classification_data
        remover = UnivariateFeatureRemover()
        remover.fit(X, y)

        assert remover.feature_importance is not None
        assert "features" in remover.feature_importance.columns
        assert "p_values" in remover.feature_importance.columns

    def test_fit_returns_self(self, classification_data):
        X, y = classification_data
        remover = UnivariateFeatureRemover()
        result = remover.fit(X, y)

        assert result is remover

    def test_high_threshold_keeps_all(self, classification_data):
        X, y = classification_data
        remover = UnivariateFeatureRemover(threshold=1.0)
        remover.fit(X, y)
        result = remover.transform(X)

        # p_value <= 1.0 always → nothing removed
        assert len(result.columns) == len(X.columns)


# ---------------------------------------------------------------------------
# InfinityHandler
# ---------------------------------------------------------------------------


class TestInfinityHandler:
    def test_replaces_positive_infinity(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
        handler = InfinityHandler(def_val=-100)
        result = handler.fit_transform(df)

        assert result["a"].iloc[1] == -100

    def test_does_not_modify_finite_values(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        handler = InfinityHandler(def_val=-100)
        result = handler.fit_transform(df)

        assert list(result["a"]) == [1.0, 2.0, 3.0]

    def test_skips_object_columns(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": ["x", "y", "z"]})
        handler = InfinityHandler(def_val=-100)
        result = handler.fit_transform(df)

        assert list(result["b"]) == ["x", "y", "z"]

    def test_multiple_inf_columns(self):
        df = pd.DataFrame({"a": [np.inf, 2.0], "b": [1.0, np.inf]})
        handler = InfinityHandler(def_val=0)
        result = handler.fit_transform(df)

        assert result["a"].iloc[0] == 0
        assert result["b"].iloc[1] == 0

    def test_default_replacement_value_is_minus_100(self):
        handler = InfinityHandler()
        assert handler.def_val == -100

    def test_fit_returns_self(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        handler = InfinityHandler()
        result = handler.fit(df)

        assert result is handler


# ---------------------------------------------------------------------------
# VotingModel
# ---------------------------------------------------------------------------


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
        estimators = [_ConstantEstimator(0.2), _ConstantEstimator(0.4)]
        model = VotingModel(estimators)
        result = model.predict(X)

        np.testing.assert_allclose(result, [0.3, 0.3, 0.3])

    def test_predict_proba_averages_estimators(self):
        X = pd.DataFrame({"a": [1, 2]})
        estimators = [_ConstantEstimator(0.0), _ConstantEstimator(1.0)]
        model = VotingModel(estimators)
        result = model.predict_proba(X)

        # Average of [1,0] and [0,1] → [0.5, 0.5]
        np.testing.assert_allclose(result[:, 0], [0.5, 0.5])
        np.testing.assert_allclose(result[:, 1], [0.5, 0.5])

    def test_single_estimator_returns_its_predictions(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        estimators = [_ConstantEstimator(0.7)]
        model = VotingModel(estimators)
        result = model.predict(X)

        np.testing.assert_allclose(result, [0.7, 0.7, 0.7])

    def test_fit_returns_self(self):
        X = pd.DataFrame({"a": [1, 2]})
        model = VotingModel([_ConstantEstimator(0.5)])
        result = model.fit(X)

        assert result is model
