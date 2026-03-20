"""Tests for stability metrics: psi and psi_from_distribution."""

import numpy as np
import pytest
from src.caketool.metric.stability_metric import psi, psi_from_distribution


class TestPsiFromDistribution:
    def test_identical_distributions_returns_near_zero(self):
        dist = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        result = psi_from_distribution(dist, dist.copy())

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_very_different_distributions_returns_high_psi(self):
        expected = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        actual = np.array([0.05, 0.05, 0.1, 0.3, 0.5])
        result = psi_from_distribution(expected, actual)

        # PSI > 0.2 indicates significant shift
        assert result > 0.2

    def test_slightly_different_distributions_returns_low_psi(self):
        expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        actual = np.array([0.21, 0.19, 0.20, 0.21, 0.19])
        result = psi_from_distribution(expected, actual)

        # Small shift → PSI < 0.1
        assert result < 0.1

    def test_returns_float(self):
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        result = psi_from_distribution(dist, dist.copy())

        assert isinstance(result, float)

    def test_unnormalized_distributions_are_normalized(self):
        # Both are multiples of the same distribution
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        actual = np.array([2.0, 4.0, 6.0, 8.0])
        result = psi_from_distribution(expected, actual)

        # After normalization they are equal → PSI ≈ 0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_zero_bins_are_clipped_not_nan(self):
        expected = np.array([0.0, 0.5, 0.5])
        actual = np.array([0.5, 0.5, 0.0])
        result = psi_from_distribution(expected, actual)

        assert np.isfinite(result)


class TestPsi:
    def test_identical_arrays_returns_near_zero(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        result = psi(data, data.copy())

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_shifted_distribution_returns_high_psi(self):
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(5, 1, 1000)  # large mean shift
        result = psi(expected, actual)

        assert result > 0.2

    def test_similar_distributions_returns_low_psi(self):
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        result = psi(expected, actual)

        # Both drawn from same distribution → PSI should be low
        assert result < 0.1

    def test_quantiles_bucket_type(self):
        np.random.seed(0)
        expected = np.random.normal(0, 1, 500)
        actual = np.random.normal(0, 1, 500)
        result = psi(expected, actual, bucket_type="quantiles", n_bins=10)

        assert np.isfinite(result)
        assert result >= 0

    def test_bins_bucket_type(self):
        np.random.seed(0)
        expected = np.random.normal(0, 1, 500)
        actual = np.random.normal(0, 1, 500)
        result = psi(expected, actual, bucket_type="bins", n_bins=10)

        assert np.isfinite(result)
        assert result >= 0

    def test_bins_and_quantiles_give_similar_results_for_stable_data(self):
        np.random.seed(1)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)

        psi_bins = psi(expected, actual, bucket_type="bins")
        psi_quantiles = psi(expected, actual, bucket_type="quantiles")

        # Both should be small and roughly comparable
        assert psi_bins < 0.15
        assert psi_quantiles < 0.15

    def test_returns_float(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = psi(data, data.copy())

        assert isinstance(result, float)

    def test_custom_n_bins(self):
        np.random.seed(42)
        expected = np.random.normal(0, 1, 500)
        actual = np.random.normal(0, 1, 500)

        result_5 = psi(expected, actual, n_bins=5)
        result_20 = psi(expected, actual, n_bins=20)

        assert np.isfinite(result_5)
        assert np.isfinite(result_20)
