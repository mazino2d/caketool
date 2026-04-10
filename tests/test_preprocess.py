"""Tests for caketool.model.preprocess transformers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from src.caketool.model.preprocess import (
    ColinearFeatureRemover,
    FeatureRemover,
    InfinityHandler,
    MissingValueImputer,
    OutlierClipper,
    UnivariateFeatureRemover,
)

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
        remover = FeatureRemover(dropped_cols=("a", "b"))
        result = remover.fit_transform(sample_df)

        assert "a" not in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_ignores_nonexistent_columns(self, sample_df):
        remover = FeatureRemover(dropped_cols=("nonexistent",))
        result = remover.fit_transform(sample_df)

        assert list(result.columns) == list(sample_df.columns)

    def test_empty_dropped_cols(self, sample_df):
        remover = FeatureRemover(dropped_cols=())
        result = remover.fit_transform(sample_df)

        assert list(result.columns) == list(sample_df.columns)

    def test_remove_all_columns(self, sample_df):
        remover = FeatureRemover(dropped_cols=("a", "b", "c"))
        result = remover.fit_transform(sample_df)

        assert len(result.columns) == 0

    def test_fit_returns_self(self, sample_df):
        remover = FeatureRemover(dropped_cols=("a",))
        assert remover.fit(sample_df) is remover

    def test_transform_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        remover = FeatureRemover(dropped_cols=("a",))
        remover.fit_transform(sample_df)

        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# ColinearFeatureRemover
# ---------------------------------------------------------------------------


class TestColinearFeatureRemover:
    def test_removes_highly_correlated_feature(self):
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

        assert len(result.columns) < len(df.columns)
        assert "c" in result.columns

    def test_removes_negatively_correlated_feature(self):
        # b = -a, so corr(a, b) = -1.0 — should still be caught
        arr = np.arange(1.0, 11.0)
        df = pd.DataFrame({"a": arr, "b": -arr, "c": np.random.default_rng(0).random(10)})
        y = pd.Series(arr > 5, dtype=int)

        remover = ColinearFeatureRemover(correlation_threshold=0.9)
        remover.fit(df, y)
        result = remover.transform(df)

        assert len(result.columns) < len(df.columns)

    def test_keeps_uncorrelated_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.random(50), "y": rng.random(50), "z": rng.random(50)})
        target = pd.Series(rng.integers(0, 2, 50))

        remover = ColinearFeatureRemover(correlation_threshold=0.9)
        remover.fit(df, target)
        result = remover.transform(df)

        assert len(result.columns) == 3

    def test_default_threshold_is_09(self):
        assert ColinearFeatureRemover().correlation_threshold == 0.9

    def test_fit_returns_self(self, sample_df):
        y = pd.Series([0, 1, 0, 1, 0])
        remover = ColinearFeatureRemover()
        assert remover.fit(sample_df, y) is remover


# ---------------------------------------------------------------------------
# UnivariateFeatureRemover
# ---------------------------------------------------------------------------


class TestUnivariateFeatureRemover:
    def test_removes_irrelevant_features(self, classification_data):
        X, y = classification_data
        X = X.copy()
        np.random.seed(0)
        X["noise"] = np.random.rand(len(X))

        remover = UnivariateFeatureRemover(threshold=0.05)
        remover.fit(X, y)
        result = remover.transform(X)

        assert "noise" not in result.columns

    def test_keeps_relevant_features(self, classification_data):
        X, y = classification_data
        remover = UnivariateFeatureRemover(threshold=0.05)
        remover.fit(X, y)

        assert len(remover.transform(X).columns) > 0

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
        assert remover.fit(X, y) is remover

    def test_high_threshold_keeps_all(self, classification_data):
        X, y = classification_data
        remover = UnivariateFeatureRemover(threshold=1.0)
        remover.fit(X, y)

        assert len(remover.transform(X).columns) == len(X.columns)


# ---------------------------------------------------------------------------
# InfinityHandler
# ---------------------------------------------------------------------------


class TestInfinityHandler:
    def test_replaces_positive_infinity(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        result = InfinityHandler(def_val=-100).fit_transform(df)

        assert result["a"].iloc[1] == -100

    def test_replaces_negative_infinity(self):
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
        result = InfinityHandler(def_val=-100).fit_transform(df)

        assert result["a"].iloc[1] == -100

    def test_does_not_modify_finite_values(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = InfinityHandler(def_val=-100).fit_transform(df)

        assert list(result["a"]) == [1.0, 2.0, 3.0]

    def test_skips_object_columns(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": ["x", "y", "z"]})
        result = InfinityHandler(def_val=-100).fit_transform(df)

        assert list(result["b"]) == ["x", "y", "z"]

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [np.inf, 2.0]})
        InfinityHandler().fit_transform(df)

        assert np.isinf(df["a"].iloc[0])

    def test_default_replacement_value_is_minus_100(self):
        assert InfinityHandler().def_val == -100

    def test_fit_returns_self(self):
        handler = InfinityHandler()
        assert handler.fit(pd.DataFrame({"a": [1.0]})) is handler


# ---------------------------------------------------------------------------
# MissingValueImputer
# ---------------------------------------------------------------------------


class TestMissingValueImputer:
    def test_fills_nan_with_median(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})
        imputer = MissingValueImputer(strategy="median")
        result = imputer.fit_transform(df)

        assert not result["a"].isna().any()
        assert result["a"].iloc[2] == pytest.approx(2.0)  # median of [1,2,4]

    def test_fills_nan_with_mean(self):
        df = pd.DataFrame({"a": [1.0, 3.0, np.nan]})
        imputer = MissingValueImputer(strategy="mean")
        result = imputer.fit_transform(df)

        assert result["a"].iloc[2] == pytest.approx(2.0)

    def test_fills_nan_with_constant(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        imputer = MissingValueImputer(strategy="constant", fill_value=-999)
        result = imputer.fit_transform(df)

        assert result["a"].iloc[1] == -999

    def test_skips_object_columns(self):
        df = pd.DataFrame({"a": [np.nan, 2.0], "b": [None, "x"]})
        imputer = MissingValueImputer(strategy="constant", fill_value=0)
        result = imputer.fit_transform(df)

        assert result["b"].iloc[0] is None

    def test_fit_returns_self(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        imputer = MissingValueImputer()
        assert imputer.fit(df) is imputer


# ---------------------------------------------------------------------------
# OutlierClipper
# ---------------------------------------------------------------------------


class TestOutlierClipper:
    def test_clips_upper_outliers(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 1000.0]})
        clipper = OutlierClipper(lower_quantile=0.0, upper_quantile=0.8)
        result = clipper.fit_transform(df)

        assert result["a"].max() <= df["a"].quantile(0.8) + 1e-9

    def test_clips_lower_outliers(self):
        df = pd.DataFrame({"a": [-1000.0, 2.0, 3.0, 4.0, 5.0]})
        clipper = OutlierClipper(lower_quantile=0.2, upper_quantile=1.0)
        result = clipper.fit_transform(df)

        assert result["a"].min() >= df["a"].quantile(0.2) - 1e-9

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, 1000.0]})
        OutlierClipper().fit_transform(df)

        assert df["a"].iloc[1] == 1000.0

    def test_skips_object_columns(self):
        df = pd.DataFrame({"a": [1.0, 100.0], "b": ["x", "y"]})
        clipper = OutlierClipper()
        result = clipper.fit_transform(df)

        assert list(result["b"]) == ["x", "y"]

    def test_fit_returns_self(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        clipper = OutlierClipper()
        assert clipper.fit(df) is clipper
