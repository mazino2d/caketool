"""Tests for risk report: decribe_risk_score."""

import numpy as np
import pandas as pd
import pytest
from src.caketool.report.risk_report import decribe_risk_score


@pytest.fixture
def score_df():
    np.random.seed(42)
    n = 500
    scores = np.random.uniform(0, 1, n)
    labels = (scores > 0.7).astype(int)  # higher score → more likely default
    return pd.DataFrame({"score": scores, "label": labels})


class TestDecriberiskScore:
    def test_returns_dataframe(self, score_df):
        result = decribe_risk_score(score_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, score_df):
        result = decribe_risk_score(score_df)
        expected_cols = [
            "band",
            "proba_segment",
            "def_rate%",
            "approval_rate%",
            "%bad_cumsum",
            "%good_cumsum",
            "%bad",
            "%good",
            "client_cumsum",
            "bad_cumsum",
            "good_cumsum",
            "client",
            "bad",
            "good",
        ]
        assert list(result.columns) == expected_cols

    def test_band_labels_are_descending(self, score_df):
        result = decribe_risk_score(score_df)
        # Band format: B<N> where N descends from high to low
        band_numbers = [int(b[1:]) for b in result["band"]]
        assert band_numbers == sorted(band_numbers, reverse=True)

    def test_total_clients_equals_input_size(self, score_df):
        result = decribe_risk_score(score_df)
        assert result["client"].sum() == len(score_df)

    def test_total_bad_equals_sum_of_labels(self, score_df):
        result = decribe_risk_score(score_df)
        assert result["bad"].sum() == score_df["label"].sum()

    def test_total_good_equals_non_default_count(self, score_df):
        result = decribe_risk_score(score_df)
        assert result["good"].sum() == (score_df["label"] == 0).sum()

    def test_approval_rate_is_cumulative(self, score_df):
        result = decribe_risk_score(score_df)
        # approval_rate% should be monotonically increasing
        rates = result["approval_rate%"].values
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_last_approval_rate_is_100(self, score_df):
        result = decribe_risk_score(score_df)
        assert result["approval_rate%"].iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_custom_pred_and_label_cols(self):
        df = pd.DataFrame(
            {
                "probability": [0.05, 0.15, 0.5, 0.75, 0.9],
                "default": [0, 0, 0, 1, 1],
            }
        )
        result = decribe_risk_score(df, pred_col="probability", label_col="default")
        assert result["client"].sum() == 5
        assert result["bad"].sum() == 2

    def test_custom_segments(self, score_df):
        custom_segments = (0, 0.25, 0.5, 0.75, 1.0)
        result = decribe_risk_score(score_df, segments=custom_segments)
        # 4 bins from 4 intervals
        assert len(result) == len(custom_segments) - 1

    def test_default_segments_produce_20_bands(self, score_df):
        result = decribe_risk_score(score_df)
        # Default segments has 21 breakpoints → 20 bands
        assert len(result) == 20

    def test_percent_bad_sums_to_100(self, score_df):
        result = decribe_risk_score(score_df)
        assert result["%bad"].sum() == pytest.approx(100.0, abs=0.01)

    def test_percent_good_sums_to_100(self, score_df):
        result = decribe_risk_score(score_df)
        assert result["%good"].sum() == pytest.approx(100.0, abs=0.01)
