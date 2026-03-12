import numpy as np
import pandas as pd
from src.caketool.calibration import calibrate_score_to_normal


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
