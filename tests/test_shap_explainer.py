"""Tests for ShapExplainer."""

from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from src.caketool.explainability import ShapExplainer

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    return X_df, pd.Series(y)


@pytest.fixture(scope="module")
def fitted_model(dataset):
    X, y = dataset
    model = GradientBoostingClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def fitted_explainer(dataset, fitted_model):
    X, _ = dataset
    explainer = ShapExplainer(model=fitted_model, explainer_type="tree")
    explainer.fit(X)
    return explainer


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestShapExplainerInit:
    def test_default_attributes(self, fitted_model):
        e = ShapExplainer(model=fitted_model)
        assert e.explainer_type == "tree"
        assert e.n_background_samples == 100
        assert e.background_data is None
        assert not e._is_fitted

    def test_custom_params(self, fitted_model, dataset):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model, explainer_type="kernel", background_data=X[:50], n_background_samples=20)
        assert e.explainer_type == "kernel"
        assert e.n_background_samples == 20
        assert e.background_data is not None


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestShapExplainerFit:
    def test_fit_dataframe_input(self, dataset, fitted_model):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model, explainer_type="tree")
        e.fit(X)
        assert e._is_fitted
        assert e.shap_values_.shape == (len(X), X.shape[1])

    def test_fit_numpy_input(self, dataset, fitted_model):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model, explainer_type="tree")
        e.fit(X.values)
        assert e._is_fitted
        assert e.feature_names_ == [f"f{i}" for i in range(10)]

    def test_fit_sets_explainer(self, dataset, fitted_model):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model, explainer_type="tree")
        e.fit(X)
        assert e.explainer_ is not None

    def test_fit_returns_self(self, dataset, fitted_model):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model, explainer_type="tree")
        result = e.fit(X)
        assert result is e

    def test_kernel_without_background_raises(self, fitted_model):
        with pytest.raises(ValueError, match="background_data is required"):
            ShapExplainer(model=fitted_model, explainer_type="kernel")

    def test_invalid_explainer_type_raises(self, fitted_model):
        with pytest.raises(ValueError, match="Invalid explainer_type"):
            ShapExplainer(model=fitted_model, explainer_type="invalid")


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------


class TestGetFeatureImportance:
    def test_returns_dataframe(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert set(result.columns) == {"feature", "mean_abs_shap", "rank"}

    def test_sorted_descending(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert result["mean_abs_shap"].is_monotonic_decreasing

    def test_rank_starts_at_one(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert result["rank"].iloc[0] == 1

    def test_all_features_present(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert len(result) == len(fitted_explainer.feature_names_)

    def test_values_nonnegative(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert (result["mean_abs_shap"] >= 0).all()

    def test_raises_before_fit(self, fitted_model):
        e = ShapExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.get_feature_importance()


# ---------------------------------------------------------------------------
# get_local_explanation
# ---------------------------------------------------------------------------


class TestGetLocalExplanation:
    def test_returns_dataframe(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_local_explanation(X, row_index=0)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_local_explanation(X, row_index=0)
        assert set(result.columns) == {"feature", "feature_value", "shap_value", "abs_shap"}

    def test_sorted_by_abs_shap(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_local_explanation(X, row_index=0)
        assert result["abs_shap"].is_monotonic_decreasing

    def test_row_count_equals_n_features(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_local_explanation(X, row_index=0)
        assert len(result) == len(fitted_explainer.feature_names_)

    def test_accepts_numpy(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_local_explanation(X.values, row_index=0)
        assert len(result) == X.shape[1]

    def test_raises_index_error_out_of_bounds(self, fitted_explainer, dataset):
        X, _ = dataset
        with pytest.raises(IndexError):
            fitted_explainer.get_local_explanation(X, row_index=999)

    def test_raises_before_fit(self, fitted_model, dataset):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.get_local_explanation(X)


# ---------------------------------------------------------------------------
# get_top_drivers
# ---------------------------------------------------------------------------


class TestGetTopDrivers:
    def test_returns_exactly_n_rows(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_top_drivers(X, row_index=0, n=5)
        assert len(result) == 5

    def test_direction_values_valid(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_top_drivers(X, row_index=0, n=10)
        assert set(result["direction"]).issubset({"positive", "negative"})

    def test_direction_consistent_with_shap_sign(self, fitted_explainer, dataset):
        X, _ = dataset
        result = fitted_explainer.get_top_drivers(X, row_index=0, n=10)
        for _, row in result.iterrows():
            if row["shap_value"] >= 0:
                assert row["direction"] == "positive"
            else:
                assert row["direction"] == "negative"

    def test_n_greater_than_features_returns_all(self, fitted_explainer, dataset):
        X, _ = dataset
        n_features = len(fitted_explainer.feature_names_)
        result = fitted_explainer.get_top_drivers(X, row_index=0, n=n_features + 100)
        assert len(result) == n_features


# ---------------------------------------------------------------------------
# show_* smoke tests
# ---------------------------------------------------------------------------


class TestShowMethods:
    def test_show_summary_smoke(self, fitted_explainer):
        with patch("shap.summary_plot") as mock_plot:
            fitted_explainer.show_summary(plot_type="bar", max_display=10)
            mock_plot.assert_called_once()

    def test_show_summary_beeswarm_smoke(self, fitted_explainer):
        with patch("shap.summary_plot") as mock_plot:
            fitted_explainer.show_summary(plot_type="beeswarm", max_display=10)
            mock_plot.assert_called_once()

    def test_show_waterfall_smoke(self, fitted_explainer, dataset):
        X, _ = dataset
        with patch("shap.waterfall_plot") as mock_plot:
            fitted_explainer.show_waterfall(X, row_index=0, max_display=10)
            mock_plot.assert_called_once()

    def test_show_dependence_invalid_feature_raises(self, fitted_explainer):
        with pytest.raises(ValueError, match="not found in feature_names_"):
            fitted_explainer.show_dependence("nonexistent_feature")

    def test_show_summary_raises_before_fit(self, fitted_model):
        e = ShapExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.show_summary()

    def test_show_waterfall_raises_before_fit(self, fitted_model, dataset):
        X, _ = dataset
        e = ShapExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.show_waterfall(X)

    def test_show_dependence_raises_before_fit(self, fitted_model):
        e = ShapExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.show_dependence("f0")
