import numpy as np
import pandas as pd
import pytest

from caketool.calibration import NormalScoreCalibrator, calibrate_score_to_normal


class TestCalibrateScoreToNormal:
    """Tests for calibrate_score_to_normal function."""

    def test_basic_array_input(self):
        """Test with basic numpy array input."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = calibrate_score_to_normal(scores)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(scores)
        assert np.all(result > 0) and np.all(result < 1)

    def test_single_float_input(self):
        """Test with a single float value."""
        score = 0.5
        result = calibrate_score_to_normal(score)

        # Score 0.5 should map to approximately 0.5
        assert 0.49 < result < 0.51

    def test_list_input(self):
        """Test with list input."""
        scores = [0.2, 0.5, 0.8]
        result = calibrate_score_to_normal(scores)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_tuple_input(self):
        """Test with tuple input."""
        scores = (0.2, 0.5, 0.8)
        result = calibrate_score_to_normal(scores)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        scores = pd.Series([0.2, 0.5, 0.8])
        result = calibrate_score_to_normal(scores)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_standard_mode(self):
        """Test with standardization enabled."""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        result_standard = calibrate_score_to_normal(scores, standard=True)
        result_non_standard = calibrate_score_to_normal(scores, standard=False)

        # Results should be different when standard=True
        assert not np.allclose(result_standard, result_non_standard)
        # Output should still be in (0, 1)
        assert np.all(result_standard > 0) and np.all(result_standard < 1)

    def test_extreme_values_clipped(self):
        """Test that extreme values (0 and 1) are handled properly."""
        scores = np.array([0.0, 0.5, 1.0])
        result = calibrate_score_to_normal(scores)

        # Should not produce inf or nan
        assert np.all(np.isfinite(result))
        assert np.all(result > 0) and np.all(result < 1)

    def test_output_range(self):
        """Test that output is always in (0, 1) range."""
        # Test with many random values
        np.random.seed(42)
        scores = np.random.uniform(0.01, 0.99, size=100)
        result = calibrate_score_to_normal(scores)

        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_ordering_preserved(self):
        """Test that relative ordering is preserved for monotonic input."""
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = calibrate_score_to_normal(scores)

        # Check that ordering is preserved
        assert np.all(np.diff(result) > 0)

    def test_symmetric_around_half(self):
        """Test symmetry: score 0.5 should map to approximately 0.5."""
        scores = np.array([0.3, 0.5, 0.7])
        result = calibrate_score_to_normal(scores)

        # Middle value should be close to 0.5
        assert 0.49 < result[1] < 0.51

    def test_docstring_examples(self):
        """Docstring examples match the actual output."""
        np.testing.assert_allclose(calibrate_score_to_normal([0.1, 0.5, 0.9]), [0.2173, 0.5, 0.7827], atol=1e-4)
        assert calibrate_score_to_normal(0.99) == pytest.approx(0.9110, abs=1e-4)

    def test_standard_mode_constant_scores_not_nan(self):
        """std = 0 (constant scores or a single score) must not produce NaN."""
        np.testing.assert_allclose(calibrate_score_to_normal([0.3, 0.3, 0.3], standard=True), [0.5, 0.5, 0.5])
        assert calibrate_score_to_normal(0.3, standard=True) == pytest.approx(0.5)


class TestNormalScoreCalibrator:
    """Tests for the fit/transform calibrator."""

    def test_matches_standard_mode_on_reference(self):
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        calibrator = NormalScoreCalibrator().fit(scores)
        np.testing.assert_allclose(calibrator.transform(scores), calibrate_score_to_normal(scores, standard=True))

    def test_transform_is_independent_of_batch(self):
        calibrator = NormalScoreCalibrator().fit(np.linspace(0.05, 0.95, 50))
        batch = calibrator.transform(np.array([0.2, 0.4, 0.6]))
        single = calibrator.transform(np.array([0.4]))
        assert batch[1] == pytest.approx(single[0])

    def test_constant_reference_not_nan(self):
        calibrator = NormalScoreCalibrator().fit([0.4, 0.4, 0.4])
        result = calibrator.transform([0.4, 0.6])
        assert np.all(np.isfinite(result))
        assert result[0] == pytest.approx(0.5)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            NormalScoreCalibrator().transform([0.5])

    def test_fit_transform(self):
        scores = np.array([0.2, 0.5, 0.8])
        result = NormalScoreCalibrator().fit_transform(scores)
        assert result.shape == scores.shape
        assert np.all(np.diff(result) > 0)
