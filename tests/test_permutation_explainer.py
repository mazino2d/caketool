"""Tests for PermutationExplainer and ModelExplainer."""

from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from src.caketool.explainability import ModelExplainer, PermutationExplainer

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
    explainer = PermutationExplainer(model=fitted_model)
    explainer.fit(X)
    return explainer


# ---------------------------------------------------------------------------
# ModelExplainer abstract class
# ---------------------------------------------------------------------------


class TestModelExplainerAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ModelExplainer()

    def test_permutation_explainer_is_subclass(self, fitted_model):
        e = PermutationExplainer(model=fitted_model)
        assert isinstance(e, ModelExplainer)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestPermutationExplainerInit:
    def test_default_attributes(self, fitted_model):
        e = PermutationExplainer(model=fitted_model)
        assert e.n_background_samples == 100
        assert e.background_data is None
        assert not e._is_fitted

    def test_custom_params(self, fitted_model, dataset):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model, background_data=X[:50], n_background_samples=20)
        assert e.n_background_samples == 20
        assert e.background_data is not None


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestPermutationExplainerFit:
    def test_fit_dataframe_input(self, dataset, fitted_model):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model)
        e.fit(X)
        assert e._is_fitted
        assert e.shap_values_.shape == (len(X), X.shape[1])

    def test_fit_numpy_input(self, dataset, fitted_model):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model)
        e.fit(X.values)
        assert e._is_fitted
        assert e.feature_names_ == [f"f{i}" for i in range(10)]

    def test_fit_sets_explainer(self, dataset, fitted_model):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model)
        e.fit(X)
        assert e.explainer_ is not None

    def test_fit_returns_self(self, dataset, fitted_model):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model)
        result = e.fit(X)
        assert result is e

    def test_fit_with_explicit_background(self, dataset, fitted_model):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model, background_data=X[:50])
        e.fit(X)
        assert e._is_fitted
        assert e.shap_values_.shape == (len(X), X.shape[1])


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------


class TestGetFeatureImportance:
    def test_returns_dataframe(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_explainer):
        result = fitted_explainer.get_feature_importance()
        assert set(result.columns) == {"rank", "feature", "importance_pct", "direction", "mean_abs_shap"}

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
        e = PermutationExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.get_feature_importance()


# ---------------------------------------------------------------------------
# get_local_explanation
# ---------------------------------------------------------------------------


class TestGetLocalExplanation:
    def test_returns_dataframe(self, fitted_explainer):
        result = fitted_explainer.get_local_explanation(row_index=0)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self, fitted_explainer):
        result = fitted_explainer.get_local_explanation(row_index=0)
        assert set(result.columns) == {"rank", "feature", "importance_pct", "direction", "feature_value", "shap_value"}

    def test_sorted_by_importance(self, fitted_explainer):
        result = fitted_explainer.get_local_explanation(row_index=0)
        assert result["importance_pct"].is_monotonic_decreasing

    def test_row_count_equals_n_features(self, fitted_explainer):
        result = fitted_explainer.get_local_explanation(row_index=0)
        assert len(result) == len(fitted_explainer.feature_names_)

    def test_raises_index_error_out_of_bounds(self, fitted_explainer):
        with pytest.raises(IndexError):
            fitted_explainer.get_local_explanation(row_index=999)

    def test_raises_before_fit(self, fitted_model):
        e = PermutationExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.get_local_explanation()


# ---------------------------------------------------------------------------
# show_* smoke tests
# ---------------------------------------------------------------------------


class TestShowMethods:
    def test_show_summary_beeswarm_smoke(self, fitted_explainer):
        with patch("shap.plots.beeswarm") as mock_plot:
            fitted_explainer.show_summary(max_display=10)
            mock_plot.assert_called_once()

    def test_show_waterfall_smoke(self, fitted_explainer):
        with patch("shap.waterfall_plot") as mock_plot:
            fitted_explainer.show_waterfall(row_index=0, max_display=10)
            mock_plot.assert_called_once()

    def test_show_dependence_invalid_feature_raises(self, fitted_explainer):
        with pytest.raises(ValueError, match="not found in feature_names_"):
            fitted_explainer.show_dependence("nonexistent_feature")

    def test_show_summary_raises_before_fit(self, fitted_model):
        e = PermutationExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.show_summary()

    def test_show_waterfall_raises_before_fit(self, fitted_model):
        e = PermutationExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.show_waterfall()

    def test_show_dependence_raises_before_fit(self, fitted_model):
        e = PermutationExplainer(model=fitted_model)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            e.show_dependence("f0")
