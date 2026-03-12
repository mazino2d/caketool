import numpy as np
import pandas as pd
import pytest

from caketool.monitor import AdversarialModel


@pytest.fixture
def reference_df():
    """Create reference dataset (no drift)."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(0, 1, n),
            "feature_b": np.random.normal(5, 2, n),
            "feature_c": np.random.choice(["cat", "dog", "bird"], n),
        }
    )


@pytest.fixture
def no_drift_df():
    """Create dataset with same distribution (no drift)."""
    np.random.seed(123)
    n = 500
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(0, 1, n),
            "feature_b": np.random.normal(5, 2, n),
            "feature_c": np.random.choice(["cat", "dog", "bird"], n),
        }
    )


@pytest.fixture
def drift_df():
    """Create dataset with different distribution (drift)."""
    np.random.seed(456)
    n = 500
    return pd.DataFrame(
        {
            "feature_a": np.random.normal(3, 1, n),  # shifted mean
            "feature_b": np.random.normal(10, 2, n),  # shifted mean
            "feature_c": np.random.choice(["cat", "fish", "rabbit"], n),  # different categories
        }
    )


class TestAdversarialModel:
    """Test AdversarialModel drift detection."""

    def test_init(self):
        """Test AdversarialModel initialization."""
        model = AdversarialModel()
        assert model.auc_score == -1
        assert model.model is not None
        assert model.encoder is not None

    def test_fit_no_drift(self, reference_df, no_drift_df):
        """Test fit with similar distributions (low AUC expected)."""
        model = AdversarialModel()
        model.fit(reference_df, no_drift_df)

        assert 0 <= model.auc_score <= 1
        # With no drift, AUC should be close to 0.5 (random guessing)
        assert model.auc_score < 0.7

    def test_fit_with_drift(self, reference_df, drift_df):
        """Test fit with different distributions (high AUC expected)."""
        model = AdversarialModel()
        model.fit(reference_df, drift_df)

        assert 0 <= model.auc_score <= 1
        # With drift, AUC should be high (model can distinguish)
        assert model.auc_score > 0.7

    def test_fit_with_custom_features(self, reference_df, drift_df):
        """Test fit with specific feature subset."""
        model = AdversarialModel()
        model.fit(reference_df, drift_df, features=["feature_a", "feature_b"])

        assert model.auc_score > 0.5
        assert len(model.feature_names_) == 2

    def test_feature_names_stored(self, reference_df, no_drift_df):
        """Test that feature names are stored after fit."""
        model = AdversarialModel()
        model.fit(reference_df, no_drift_df)

        assert hasattr(model, "feature_names_")
        assert len(model.feature_names_) == 3

    def test_show(self, reference_df, drift_df, capsys):
        """Test show method outputs correctly."""
        model = AdversarialModel()
        model.fit(reference_df, drift_df)
        model.show(n_features=2)

        captured = capsys.readouterr()
        assert "ROC AUC:" in captured.out
        assert "Top 2 important feature(s)" in captured.out
