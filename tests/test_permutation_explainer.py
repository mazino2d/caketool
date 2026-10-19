"""Tests for PermutationExplainer and ModelExplainer."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
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


# ---------------------------------------------------------------------------
# Model types: regressors, wide inputs, multiclass
# ---------------------------------------------------------------------------


class _RegressorWithProba(RegressorMixin, BaseEstimator):
    """Regressor exposing a ``predict_proba`` that cannot be used."""

    def fit(self, X, y):
        self.model_ = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        raise NotImplementedError("regressor")


class TestPermutationExplainerModelTypes:
    def test_regressor_uses_predict(self, dataset):
        X, y = dataset
        model = _RegressorWithProba().fit(X, y.astype(float))
        e = PermutationExplainer(model=model, n_background_samples=20).fit(X[:10])
        assert e.shap_values_.shape == (10, X.shape[1])

    def test_wide_input_above_default_max_evals(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(20, 300)), columns=[f"f{i}" for i in range(300)])
        y = (X["f0"] > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        e = PermutationExplainer(model=model, n_background_samples=5).fit(X[:2])
        assert e.shap_values_.shape == (2, 300)

    @pytest.fixture(scope="class")
    def multiclass(self):
        X, y = make_classification(n_samples=150, n_features=5, n_informative=3, n_classes=3, random_state=0)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        return X, LogisticRegression(max_iter=500).fit(X, y)

    def test_multiclass_requires_class_index(self, multiclass):
        X, model = multiclass
        with pytest.raises(ValueError, match="class_index"):
            PermutationExplainer(model=model).fit(X[:5])

    def test_multiclass_explains_selected_class(self, multiclass):
        X, model = multiclass
        e = PermutationExplainer(model=model, background_data=X, n_background_samples=150, class_index=2).fit(X[:5])
        assert e.shap_values_.shape == (5, 5)
        # Additivity: base value + SHAP values = predicted probability of the selected class
        np.testing.assert_allclose(
            e.base_value_ + e.shap_values_.sum(axis=1), model.predict_proba(X[:5])[:, 2], atol=1e-6
        )

    def test_class_index_out_of_range_raises(self, multiclass):
        X, model = multiclass
        with pytest.raises(ValueError, match="out of range"):
            PermutationExplainer(model=model, class_index=5).fit(X[:5])

    def test_binary_class_index_zero(self, dataset, fitted_model):
        X, _ = dataset
        e = PermutationExplainer(model=fitted_model, background_data=X, n_background_samples=200, class_index=0)
        e.fit(X[:5])
        np.testing.assert_allclose(
            e.base_value_ + e.shap_values_.sum(axis=1), fitted_model.predict_proba(X[:5])[:, 0], atol=1e-6
        )
