"""Tests for caketool.model out-of-fold training."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from caketool.model import ColinearFeatureSelector, OOFEnsemble, UnivariateFeatureSelector, fit_oof


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=300, n_features=8, random_state=0)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)], index=range(1000, 1300)), pd.Series(y)


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=0)
    return X, y


class TestFitOOF:
    def test_predictions_aligned_to_rows(self, binary_data):
        X, y = binary_data
        result = fit_oof(LogisticRegression(), X, y, cv=5)

        assert result.oof_predictions.shape == (len(X), 2)
        assert not np.isnan(result.oof_predictions).any()
        assert len(result.estimators) == len(result.folds) == 5

    def test_each_row_predicted_by_model_that_did_not_see_it(self, binary_data):
        X, y = binary_data
        result = fit_oof(LogisticRegression(), X, y, cv=3)

        for model, (train_idx, val_idx) in zip(result.estimators, result.folds, strict=True):
            assert set(train_idx).isdisjoint(val_idx)
            expected = model.predict_proba(X.iloc[val_idx])
            np.testing.assert_allclose(result.oof_predictions[val_idx], expected)

    def test_repeats_average_predictions(self, binary_data):
        X, y = binary_data
        result = fit_oof(LogisticRegression(), X, y, cv=3, n_repeats=2)

        assert len(result.estimators) == 6
        np.testing.assert_allclose(result.oof_predictions.sum(axis=1), 1.0)

    def test_regressor_uses_predict(self, regression_data):
        X, y = regression_data
        result = fit_oof(LinearRegression(), X, y, cv=4)

        assert result.oof_predictions.shape == (len(X),)
        assert np.corrcoef(result.oof_predictions, y)[0, 1] > 0.99

    def test_rows_never_held_out_are_nan(self, regression_data):
        X, y = regression_data
        splits = [(np.arange(100, 200), np.arange(0, 50))]

        class FixedSplit:
            def split(self, X, y=None, groups=None):
                return iter(splits)

        result = fit_oof(LinearRegression(), X, y, cv=FixedSplit())
        assert not np.isnan(result.oof_predictions[:50]).any()
        assert np.isnan(result.oof_predictions[50:]).all()

    def test_groups_never_span_folds(self, binary_data):
        X, y = binary_data
        groups = np.repeat(np.arange(30), 10)
        result = fit_oof(LogisticRegression(), X, y, cv=5, groups=groups)

        for train_idx, val_idx in result.folds:
            assert set(groups[train_idx]).isdisjoint(groups[val_idx])

    def test_groups_with_repeats_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="n_repeats"):
            fit_oof(LogisticRegression(), X, y, n_repeats=2, groups=np.zeros(len(X)))

    def test_custom_splitter(self, regression_data):
        X, y = regression_data
        result = fit_oof(LinearRegression(), X, y, cv=KFold(n_splits=2))

        assert len(result.estimators) == 2

    def test_length_mismatch_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="different lengths"):
            fit_oof(LogisticRegression(), X, y[:-1])

    def test_estimator_is_cloned(self, binary_data):
        X, y = binary_data
        estimator = LogisticRegression()
        result = fit_oof(estimator, X, y, cv=2)

        assert not hasattr(estimator, "coef_")
        assert result.estimators[0] is not result.estimators[1]

    def test_reproducible(self, binary_data):
        X, y = binary_data
        first = fit_oof(LogisticRegression(), X, y, cv=3, random_state=1).oof_predictions
        second = fit_oof(LogisticRegression(), X, y, cv=3, random_state=1).oof_predictions

        np.testing.assert_array_equal(first, second)

    def test_sample_weight_sliced_per_fold(self, binary_data):
        X, y = binary_data
        weights = np.ones(len(X))
        result = fit_oof(LogisticRegression(), X, y, cv=3, sample_weight=weights)

        np.testing.assert_allclose(
            result.oof_predictions, fit_oof(LogisticRegression(), X, y, cv=3).oof_predictions, rtol=1e-6
        )

    def test_early_stopping_uses_split_of_training_fold(self, binary_data):
        X, y = binary_data

        class Recorder(LogisticRegression):
            def fit(self, X, y, eval_set=None, sample_weight=None):
                self.fit_index_ = X.index
                self.es_index_ = eval_set[0][0].index
                return super().fit(X, y, sample_weight=sample_weight)

        result = fit_oof(Recorder(), X, y, cv=3, early_stopping_size=0.25)

        for model, (train_idx, val_idx) in zip(result.estimators, result.folds, strict=True):
            train_rows, val_rows = set(X.index[train_idx]), set(X.index[val_idx])
            assert set(model.fit_index_) | set(model.es_index_) == train_rows
            assert set(model.fit_index_).isdisjoint(model.es_index_)
            assert val_rows.isdisjoint(model.es_index_)
            assert len(model.es_index_) == 50

    def test_early_stopping_with_xgboost(self, binary_data):
        X, y = binary_data
        result = fit_oof(
            XGBClassifier(n_estimators=500, learning_rate=0.3, early_stopping_rounds=5),
            X,
            y,
            cv=3,
            early_stopping_size=0.25,
            fit_params={"verbose": False},
        )

        assert all(model.best_iteration < 499 for model in result.estimators)

    def test_pipeline_with_selectors(self, binary_data):
        X, y = binary_data
        pipeline = make_pipeline(UnivariateFeatureSelector(), ColinearFeatureSelector(), LogisticRegression())
        result = fit_oof(pipeline, X, y, cv=3)

        assert result.oof_predictions.shape == (len(X), 2)


class TestOOFEnsemble:
    def test_predict_proba_is_mean(self, binary_data):
        X, y = binary_data
        result = fit_oof(LogisticRegression(), X, y, cv=3)
        expected = np.mean([m.predict_proba(X) for m in result.estimators], axis=0)

        np.testing.assert_allclose(result.ensemble.predict_proba(X), expected)

    def test_predict_returns_classes(self, binary_data):
        X, y = binary_data
        ensemble = fit_oof(LogisticRegression(), X, y, cv=3).ensemble

        assert set(ensemble.predict(X)) <= set(ensemble.classes_)

    def test_regressor_predict_is_mean(self, regression_data):
        X, y = regression_data
        result = fit_oof(LinearRegression(), X, y, cv=3)
        expected = np.mean([m.predict(X) for m in result.estimators], axis=0)

        np.testing.assert_allclose(result.ensemble.predict(X), expected)

    def test_is_ensemble_of_fold_estimators(self, binary_data):
        X, y = binary_data
        result = fit_oof(LogisticRegression(), X, y, cv=2)

        assert isinstance(result.ensemble, OOFEnsemble)
        assert result.ensemble.estimators is result.estimators
