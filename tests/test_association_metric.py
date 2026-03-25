"""Tests for the association metric module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.caketool.metric.association_metric import association


@pytest.fixture()
def num_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "age": rng.integers(20, 60, n).astype(float),
            "income": rng.normal(50_000, 15_000, n),
            "edu": rng.choice(["HS", "BS", "MS", "PhD"], n),
            "job": rng.choice(["A", "B", "C"], n),
        }
    )


# ===========================================================================
# association() — pearson
# ===========================================================================


class TestAssociationPearson:
    def test_returns_tuple_of_two_floats(self, num_df):
        result = association(num_df["age"], num_df["income"], method="pearson")
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_correlation_in_range(self, num_df):
        r, _ = association(num_df["age"], num_df["income"], method="pearson")
        assert -1.0 <= r <= 1.0

    def test_p_value_in_range(self, num_df):
        _, p = association(num_df["age"], num_df["income"], method="pearson")
        assert 0.0 <= p <= 1.0

    def test_perfect_positive_correlation(self):
        x = np.arange(10, dtype=float)
        r, p = association(x, x, method="pearson")
        assert r == pytest.approx(1.0)
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        x = np.arange(10, dtype=float)
        r, p = association(x, -x, method="pearson")
        assert r == pytest.approx(-1.0)
        assert p == pytest.approx(0.0, abs=1e-6)

    def test_accepts_numpy_arrays(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r, p = association(x, x * 2, method="pearson")
        assert r == pytest.approx(1.0)


# ===========================================================================
# association() — spearman
# ===========================================================================


class TestAssociationSpearman:
    def test_returns_tuple_of_two_floats(self, num_df):
        result = association(num_df["age"], num_df["income"], method="spearman")
        assert isinstance(result, tuple) and len(result) == 2

    def test_correlation_in_range(self, num_df):
        r, _ = association(num_df["age"], num_df["income"], method="spearman")
        assert -1.0 <= r <= 1.0

    def test_monotonic_relationship_high_corr(self):
        x = np.arange(100, dtype=float)
        r, _ = association(x, x**2, method="spearman")
        assert r == pytest.approx(1.0)


# ===========================================================================
# association() — eta
# ===========================================================================


class TestAssociationEta:
    def test_returns_tuple_of_two_floats(self, num_df):
        result = association(num_df["edu"], num_df["income"], method="eta")
        assert isinstance(result, tuple) and len(result) == 2

    def test_eta_in_range(self, num_df):
        eta, _ = association(num_df["edu"], num_df["income"], method="eta")
        assert 0.0 <= eta <= 1.0

    def test_p_value_in_range(self, num_df):
        _, p = association(num_df["edu"], num_df["income"], method="eta")
        assert 0.0 <= p <= 1.0

    def test_high_eta_when_groups_differ(self):
        cat = pd.Series(["A"] * 50 + ["B"] * 50)
        num = pd.Series([1.0] * 50 + [100.0] * 50)
        eta, p = association(cat, num, method="eta")
        assert eta > 0.9
        assert p < 0.05

    def test_zero_eta_when_identical_groups(self):
        cat = pd.Series(["A"] * 50 + ["B"] * 50)
        num = pd.Series([5.0] * 100)
        eta, _ = association(cat, num, method="eta")
        assert eta == pytest.approx(0.0)


# ===========================================================================
# association() — cramers_v
# ===========================================================================


class TestAssociationCramersV:
    def test_returns_tuple_of_two_floats(self, num_df):
        result = association(num_df["edu"], num_df["job"], method="cramers_v")
        assert isinstance(result, tuple) and len(result) == 2

    def test_v_in_range(self, num_df):
        v, _ = association(num_df["edu"], num_df["job"], method="cramers_v")
        assert 0.0 <= v <= 1.0

    def test_p_value_in_range(self, num_df):
        _, p = association(num_df["edu"], num_df["job"], method="cramers_v")
        assert 0.0 <= p <= 1.0

    def test_perfect_association(self):
        cat = pd.Series(["A"] * 50 + ["B"] * 50)
        v, p = association(cat, cat, method="cramers_v")
        assert v > 0.95  # Yates' continuity correction on 2×2 table prevents exact 1.0
        assert p < 0.05

    def test_independent_columns_low_v(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.choice(["X", "Y"], 500))
        b = pd.Series(rng.choice(["P", "Q"], 500))
        v, _ = association(a, b, method="cramers_v")
        assert v < 0.15


# ===========================================================================
# association() — error handling
# ===========================================================================


class TestAssociationErrors:
    def test_invalid_method_raises(self, num_df):
        with pytest.raises(ValueError, match="method must be one of"):
            association(num_df["age"], num_df["income"], method="kendall")
