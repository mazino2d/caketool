"""Tests for BinaryBoostTree, MulticlassBoostTree, and BaseEnsemble."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from src.caketool.model.base.ensemble import BaseEnsemble
from src.caketool.model.classification.binary import BinaryBoostTree
from src.caketool.model.classification.multiclass import MulticlassBoostTree
from src.caketool.model.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=300, n_features=10, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def multiclass_data():
    X, y = make_classification(n_samples=300, n_features=10, n_classes=3, n_informative=5, random_state=42)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), pd.Series(y)


@pytest.fixture
def fast_config():
    """Minimal config for fast test runs."""
    return ModelConfig(n_estimators=10, max_depth=3)


# ---------------------------------------------------------------------------
# BinaryBoostTree
# ---------------------------------------------------------------------------


class TestBinaryBoostTree:
    def test_fit_predict_proba_shape(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fit_predict_shape(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)

        assert model.predict(X).shape == (len(X),)

    def test_classes_set_after_fit(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)

        np.testing.assert_array_equal(model.classes_, [0, 1])

    def test_input_features_stored(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)

        assert model.input_features_ == list(X.columns)

    def test_required_input_features_subset_of_input(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)

        required = set(model.get_required_input_features())
        assert required.issubset(set(X.columns))
        assert len(required) > 0

    def test_feature_schema_keys(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)

        schema = model.feature_schema_
        for key in ("input", "required", "dropped_by_univariate", "dropped_by_colinear", "model_features"):
            assert key in schema

    def test_predict_with_required_features_only(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)

        required = model.get_required_input_features()
        proba_full = model.predict_proba(X)
        proba_subset = model.predict_proba(X[required])

        np.testing.assert_allclose(proba_full, proba_subset, atol=1e-6)

    def test_get_feature_importance_columns(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)
        fi = model.get_feature_importance()

        expected_cols = {
            "feature_name",
            "gain",
            "cover",
            "total_gain",
            "total_cover",
            "weight",
            "gain_pct",
            "cover_pct",
            "total_gain_pct",
            "total_cover_pct",
            "weight_pct",
        }
        assert expected_cols.issubset(set(fi.columns))

    def test_pct_columns_sum_to_one(self, binary_data, fast_config):
        X, y = binary_data
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)
        fi = model.get_feature_importance()

        for col in ["gain_pct", "cover_pct", "total_gain_pct", "total_cover_pct", "weight_pct"]:
            assert abs(fi[col].sum() - 1.0) < 1e-5

    def test_eval_set_does_not_raise(self, binary_data, fast_config):
        X, y = binary_data
        X_tr, X_val = X.iloc[:200], X.iloc[200:]
        y_tr, y_val = y.iloc[:200], y.iloc[200:]
        model = BinaryBoostTree(fast_config)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        assert model.predict_proba(X_val).shape == (len(X_val), 2)

    def test_fit_oof_returns_correct_shapes(self, binary_data, fast_config):
        X, y = binary_data
        models, oof_pred, oof_labels = BinaryBoostTree.fit_oof(X, y, config=fast_config, n_splits=3)

        assert len(models) == 3
        assert oof_pred.shape[0] == len(y)
        assert oof_labels.shape[0] == len(y)

    def test_fit_oof_callable_on_instance(self, binary_data, fast_config):
        X, y = binary_data
        instance = BinaryBoostTree(fast_config)
        instance.fit(X, y)
        # classmethod call on instance must NOT silently bind self → X
        models, oof_pred, _ = instance.fit_oof(X, y, config=fast_config, n_splits=3)

        assert oof_pred.shape[0] == len(y)

    def test_with_categorical_features(self):
        # Use threshold=1.0 so no features are dropped by univariate remover
        # (random labels would make all features statistically insignificant)
        cfg = ModelConfig(n_estimators=10, max_depth=3, univariate_threshold=1.0)
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "num1": rng.standard_normal(100),
                "num2": rng.standard_normal(100),
                "cat1": rng.choice(["A", "B", "C"], 100),
            }
        )
        y = pd.Series(rng.integers(0, 2, 100))
        model = BinaryBoostTree(cfg)
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_no_mutation_of_input(self, binary_data, fast_config):
        X, y = binary_data
        X_copy = X.copy()
        model = BinaryBoostTree(fast_config)
        model.fit(X, y)
        model.predict_proba(X)

        pd.testing.assert_frame_equal(X, X_copy)

    def test_fit_with_missing_values(self, binary_data, fast_config):
        X, y = binary_data
        X = X.copy()
        X.loc[::7, "f0"] = np.nan
        model = BinaryBoostTree(fast_config).fit(X, y)

        assert model.predict_proba(X).shape == (len(X), 2)

    def test_fit_with_infinite_values(self, binary_data, fast_config):
        X, y = binary_data
        X = X.copy()
        X.loc[:10, "f1"] = np.inf
        X.loc[11:20, "f1"] = -np.inf
        cfg = ModelConfig(n_estimators=10, max_depth=3, use_outlier_clipper=True)
        model = BinaryBoostTree(cfg).fit(X, y)

        lower, upper = model.preprocess.named_steps["outlier_clipper"].bounds_["f1"]
        assert np.isfinite(lower) and np.isfinite(upper)
        assert np.isfinite(model.predict_proba(X)).all()

    def test_fit_with_numpy_target_and_custom_index(self, binary_data, fast_config):
        X, y = binary_data
        X = X.set_index(pd.Index(range(1000, 1000 + len(X))))
        model = BinaryBoostTree(fast_config).fit(X, y.to_numpy())

        assert model.predict(X).shape == (len(X),)

    def test_fit_with_bool_and_category_columns(self, binary_data, fast_config):
        X, y = binary_data
        X = X.assign(flag=X["f0"] > 0, grade=pd.Categorical(np.where(X["f1"] > 0, "A", "B")))
        cfg = ModelConfig(n_estimators=10, max_depth=3, use_outlier_clipper=True)
        model = BinaryBoostTree(cfg).fit(X, y)

        assert model.predict_proba(X).shape == (len(X), 2)

    def test_predict_returns_original_labels(self, binary_data, fast_config):
        X, y = binary_data
        labels = pd.Series(np.where(y == 1, "bad", "good"))
        model = BinaryBoostTree(fast_config).fit(X, labels)

        assert list(model.classes_) == ["bad", "good"]
        assert set(model.predict(X)) <= {"bad", "good"}

    def test_dropped_categorical_not_required_and_no_warning(self, binary_data, fast_config):
        X, y = binary_data
        rng = np.random.default_rng(0)
        X = X.assign(noise_cat=rng.choice(list("xyz"), len(X)))
        model = BinaryBoostTree(fast_config).fit(X, y)

        assert "noise_cat" not in model.required_input_features_
        # Third-party deprecation notices (e.g. category_encoders on pandas 3)
        # are not what this test is about; the encoder's UserWarnings are.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            model.predict_proba(X[model.required_input_features_])

    def test_onehot_lineage_keeps_source_column_required(self, binary_data):
        X, y = binary_data
        X = X.assign(city=np.where(X["f0"] > 0, "hcm", "hn"))
        cfg = ModelConfig(n_estimators=10, max_depth=3, encoder_name="category_encoders.OneHotEncoder")
        model = BinaryBoostTree(cfg).fit(X, y)

        uses_city = any(f.startswith("city_") for f in model.get_feature_names())
        assert ("city" in model.required_input_features_) == uses_city
        model.predict_proba(X[model.required_input_features_])

    def test_target_encoding_is_out_of_fold(self, binary_data, fast_config):
        X, y = binary_data
        rng = np.random.default_rng(0)
        # High-cardinality noise: in-sample target encoding would make it look predictive.
        X = X.assign(noise_id=[f"id{i}" for i in rng.integers(0, 100, len(X))])
        model = BinaryBoostTree(fast_config).fit(X, y)

        assert "noise_id" not in model.required_input_features_

    def test_early_stopping_requires_eval_set(self, binary_data, fast_config):
        X, y = binary_data
        with pytest.raises(ValueError):
            BinaryBoostTree(fast_config).fit(X, y, early_stopping_rounds=3)

    def test_fit_oof_with_early_stopping(self, binary_data):
        X, y = binary_data
        cfg = ModelConfig(n_estimators=200, max_depth=3)
        models, oof_pred, _ = BinaryBoostTree.fit_oof(X, y, config=cfg, n_splits=3, early_stopping_rounds=5)

        assert oof_pred.shape == (len(y), 2)
        assert all(m.model.best_iteration < 199 for m in models)


# ---------------------------------------------------------------------------
# MulticlassBoostTree
# ---------------------------------------------------------------------------


class TestMulticlassBoostTree:
    def test_predict_proba_shape(self, multiclass_data, fast_config):
        X, y = multiclass_data
        model = MulticlassBoostTree(fast_config)
        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (len(X), 3)

    def test_predict_shape(self, multiclass_data, fast_config):
        X, y = multiclass_data
        model = MulticlassBoostTree(fast_config)
        model.fit(X, y)

        assert model.predict(X).shape == (len(X),)

    def test_non_contiguous_labels(self, multiclass_data, fast_config):
        X, y = multiclass_data
        labels = y.map({0: 10, 1: 20, 2: 30})
        model = MulticlassBoostTree(fast_config).fit(X, labels)

        assert list(model.classes_) == [10, 20, 30]
        assert set(model.predict(X)) <= {10, 20, 30}

    def test_num_class_mismatch_raises(self, multiclass_data, fast_config):
        X, y = multiclass_data
        with pytest.raises(ValueError):
            MulticlassBoostTree(fast_config, num_class=4).fit(X, y)


# ---------------------------------------------------------------------------
# BaseEnsemble
# ---------------------------------------------------------------------------


class TestBaseEnsemble:
    def test_predict_proba_averages(self, binary_data, fast_config):
        X, y = binary_data
        models, _, _ = BinaryBoostTree.fit_oof(X, y, config=fast_config, n_splits=3)
        ensemble = BaseEnsemble(models)
        proba = ensemble.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_required_input_features_is_union(self, binary_data, fast_config):
        X, y = binary_data
        models, _, _ = BinaryBoostTree.fit_oof(X, y, config=fast_config, n_splits=3)
        ensemble = BaseEnsemble(models)

        all_required = set()
        for m in models:
            all_required.update(m.get_required_input_features())

        assert set(ensemble.required_input_features_) == all_required

    def test_get_feature_importance_has_pct_cols(self, binary_data, fast_config):
        X, y = binary_data
        models, _, _ = BinaryBoostTree.fit_oof(X, y, config=fast_config, n_splits=3)
        ensemble = BaseEnsemble(models)
        fi = ensemble.get_feature_importance()

        for col in ["gain_pct", "cover_pct", "total_gain_pct", "total_cover_pct", "weight_pct"]:
            assert col in fi.columns

    def test_fit_returns_self(self, binary_data, fast_config):
        X, y = binary_data
        models, _, _ = BinaryBoostTree.fit_oof(X, y, config=fast_config, n_splits=3)
        ensemble = BaseEnsemble(models)

        assert ensemble.fit() is ensemble

    def test_predict_returns_class_labels(self, binary_data, fast_config):
        X, y = binary_data
        models, _, _ = BinaryBoostTree.fit_oof(X, y, config=fast_config, n_splits=3)
        ensemble = BaseEnsemble(models)
        pred = ensemble.predict(X)

        assert set(np.unique(pred)) <= {0, 1}
        np.testing.assert_array_equal(pred, np.argmax(ensemble.predict_proba(X), axis=1))
