"""Tests for caketool.model feature selectors."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline

from caketool.model import ColinearFeatureSelector, UnivariateFeatureSelector

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
# ColinearFeatureSelector
# ---------------------------------------------------------------------------


class TestColinearFeatureSelector:
    def test_removes_highly_correlated_feature(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "b": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
                "c": [10.0, 2.0, 8.0, 4.0, 6.0, 1.0, 9.0, 3.0, 7.0, 5.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        selector = ColinearFeatureSelector(correlation_threshold=0.9)
        selector.fit(df, y)
        result = selector.transform(df)

        assert len(result.columns) < len(df.columns)
        assert "c" in result.columns

    def test_removes_negatively_correlated_feature(self):
        # b = -a, so corr(a, b) = -1.0 — should still be caught
        arr = np.arange(1.0, 11.0)
        df = pd.DataFrame({"a": arr, "b": -arr, "c": np.random.default_rng(0).random(10)})
        y = pd.Series(arr > 5, dtype=int)

        selector = ColinearFeatureSelector(correlation_threshold=0.9)
        selector.fit(df, y)
        result = selector.transform(df)

        assert len(result.columns) < len(df.columns)

    def test_keeps_uncorrelated_features(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.random(50), "y": rng.random(50), "z": rng.random(50)})
        target = pd.Series(rng.integers(0, 2, 50))

        selector = ColinearFeatureSelector(correlation_threshold=0.9)
        selector.fit(df, target)
        result = selector.transform(df)

        assert len(result.columns) == 3

    def test_default_threshold_is_09(self):
        assert ColinearFeatureSelector().correlation_threshold == 0.9

    def test_fit_returns_self(self, sample_df):
        y = pd.Series([0, 1, 0, 1, 0])
        selector = ColinearFeatureSelector()
        assert selector.fit(sample_df, y) is selector

    @pytest.fixture
    def collinear_df(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal(50)
        return pd.DataFrame({"a": a, "dup": a + 0.01, "c": rng.standard_normal(50)}, index=range(100, 150)), (
            a > 0
        ).astype(int)

    def test_target_matched_by_position_not_index(self, collinear_df):
        df, y = collinear_df
        selector = ColinearFeatureSelector().fit(df, pd.Series(y))  # default RangeIndex differs from df.index

        assert selector.dropped_cols_ == ["dup"]

    def test_accepts_numpy_target(self, collinear_df):
        df, y = collinear_df
        selector = ColinearFeatureSelector().fit(df, y)

        assert selector.dropped_cols_ == ["dup"]

    def test_handles_nan(self, collinear_df):
        df, y = collinear_df
        df = df.copy()
        df.loc[df.index[::4], "a"] = np.nan
        selector = ColinearFeatureSelector().fit(df, y)

        assert len(selector.dropped_cols_) == 1


# ---------------------------------------------------------------------------
# UnivariateFeatureSelector
# ---------------------------------------------------------------------------


class TestUnivariateFeatureSelector:
    def test_removes_irrelevant_features(self, classification_data):
        X, y = classification_data
        X = X.copy()
        np.random.seed(0)
        X["noise"] = np.random.rand(len(X))

        selector = UnivariateFeatureSelector(threshold=0.05)
        selector.fit(X, y)
        result = selector.transform(X)

        assert "noise" not in result.columns

    def test_keeps_relevant_features(self, classification_data):
        X, y = classification_data
        selector = UnivariateFeatureSelector(threshold=0.05)
        selector.fit(X, y)

        assert len(selector.transform(X).columns) > 0

    def test_feature_importance_stored_after_fit(self, classification_data):
        X, y = classification_data
        selector = UnivariateFeatureSelector()
        selector.fit(X, y)

        assert selector.feature_importance_ is not None
        assert "features" in selector.feature_importance_.columns
        assert "p_values" in selector.feature_importance_.columns

    def test_fit_returns_self(self, classification_data):
        X, y = classification_data
        selector = UnivariateFeatureSelector()
        assert selector.fit(X, y) is selector

    def test_high_threshold_keeps_all(self, classification_data):
        X, y = classification_data
        selector = UnivariateFeatureSelector(threshold=1.0)
        selector.fit(X, y)

        assert len(selector.transform(X).columns) == len(X.columns)

    def test_handles_nan_by_scoring_non_missing_rows(self, classification_data):
        X, y = classification_data
        X = X.copy()
        X.loc[::5, "f0"] = np.nan
        full = UnivariateFeatureSelector().fit(X.dropna(), y[X["f0"].notna()])
        selector = UnivariateFeatureSelector().fit(X, y)

        f0 = selector.feature_importance_.set_index("features").loc["f0", "p_values"]
        expected = full.feature_importance_.set_index("features").loc["f0", "p_values"]
        assert f0 == pytest.approx(expected)

    def test_all_nan_column_is_dropped(self, classification_data):
        X, y = classification_data
        X = X.assign(empty=np.nan)
        selector = UnivariateFeatureSelector().fit(X, y)

        assert "empty" in selector.dropped_cols_

    def test_accepts_numpy_target(self, classification_data):
        X, y = classification_data
        selector = UnivariateFeatureSelector().fit(X, y.to_numpy())

        assert selector.dropped_cols_ == UnivariateFeatureSelector().fit(X, y).dropped_cols_

    def test_transform_before_fit_raises(self, classification_data):
        X, _ = classification_data
        with pytest.raises(NotFittedError):
            UnivariateFeatureSelector().transform(X)


# ---------------------------------------------------------------------------
# Shared selector behaviour
# ---------------------------------------------------------------------------


class TestSelectorOutputs:
    @pytest.mark.parametrize("selector_cls", [ColinearFeatureSelector, UnivariateFeatureSelector])
    def test_selected_cols_complement_dropped(self, selector_cls, classification_data):
        X, y = classification_data
        X = X.assign(dup=X["f0"] + 1e-6)
        selector = selector_cls().fit(X, y)

        assert selector.selected_cols_ == [c for c in X.columns if c not in selector.dropped_cols_]
        assert list(selector.get_feature_names_out()) == selector.selected_cols_
        assert list(selector.transform(X).columns) == selector.selected_cols_

    def test_colinear_dropped_cols_have_no_duplicates(self):
        # a and b are independent and both kept; c = a - b is collinear with each of them
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal(500), rng.standard_normal(500)
        X = pd.DataFrame({"a": a, "b": b, "c": a - b})
        selector = ColinearFeatureSelector(correlation_threshold=0.6).fit(X, (a + b > 0).astype(int))

        assert selector.dropped_cols_ == ["c"]

    def test_chained_in_pipeline(self, classification_data):
        X, y = classification_data
        X = X.assign(dup=X["f0"] + 1e-6)
        pipeline = make_pipeline(UnivariateFeatureSelector(threshold=1.0), ColinearFeatureSelector())
        result = pipeline.fit_transform(X, y)

        assert "dup" not in result.columns or "f0" not in result.columns
