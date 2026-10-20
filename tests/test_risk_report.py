"""Tests for risk report: describe_risk_score."""

import numpy as np
import pandas as pd
import pytest

from caketool.report.risk_report import describe_risk_score


@pytest.fixture
def score_df():
    np.random.seed(42)
    n = 500
    scores = np.random.uniform(0, 1, n)
    labels = (scores > 0.7).astype(int)  # higher score → more likely default
    return pd.DataFrame({"score": scores, "label": labels})


class TestDescribeRiskScore:
    def test_returns_dataframe(self, score_df):
        result = describe_risk_score(score_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self, score_df):
        result = describe_risk_score(score_df)
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
        result = describe_risk_score(score_df)
        # Band format: B<N> where N descends from high to low
        band_numbers = [int(b[1:]) for b in result["band"]]
        assert band_numbers == sorted(band_numbers, reverse=True)

    def test_total_clients_equals_input_size(self, score_df):
        result = describe_risk_score(score_df)
        assert result["client"].sum() == len(score_df)

    def test_total_bad_equals_sum_of_labels(self, score_df):
        result = describe_risk_score(score_df)
        assert result["bad"].sum() == score_df["label"].sum()

    def test_total_good_equals_non_default_count(self, score_df):
        result = describe_risk_score(score_df)
        assert result["good"].sum() == (score_df["label"] == 0).sum()

    def test_approval_rate_is_cumulative(self, score_df):
        result = describe_risk_score(score_df)
        # approval_rate% should be monotonically increasing
        rates = result["approval_rate%"].values
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_last_approval_rate_is_100(self, score_df):
        result = describe_risk_score(score_df)
        assert result["approval_rate%"].iloc[-1] == pytest.approx(100.0, abs=0.01)

    def test_custom_pred_and_label_cols(self):
        df = pd.DataFrame(
            {
                "probability": [0.05, 0.15, 0.5, 0.75, 0.9],
                "default": [0, 0, 0, 1, 1],
            }
        )
        result = describe_risk_score(df, pred_col="probability", label_col="default")
        assert result["client"].sum() == 5
        assert result["bad"].sum() == 2

    def test_custom_segments(self, score_df):
        custom_segments = (0, 0.25, 0.5, 0.75, 1.0)
        result = describe_risk_score(score_df, segments=custom_segments)
        # 4 bins from 4 intervals
        assert len(result) == len(custom_segments) - 1

    def test_default_segments_produce_20_bands(self, score_df):
        result = describe_risk_score(score_df)
        # Default segments has 21 breakpoints → 20 bands
        assert len(result) == 20

    def test_percent_bad_sums_to_100(self, score_df):
        result = describe_risk_score(score_df)
        assert result["%bad"].sum() == pytest.approx(100.0, abs=0.01)

    def test_percent_good_sums_to_100(self, score_df):
        result = describe_risk_score(score_df)
        assert result["%good"].sum() == pytest.approx(100.0, abs=0.01)


class TestDescribeRiskScoreEdgeCases:
    def test_missing_and_out_of_range_scores_in_missing_row(self):
        df = pd.DataFrame({"score": [0.05, 0.15, np.nan, 1.5, 0.85], "label": [0, 1, 0, 1, 0]})
        result = describe_risk_score(df, segments=(0, 0.5, 1))
        assert list(result["band"]) == ["B2", "B1", "MISSING"]
        assert result["client"].sum() == len(df)
        assert result["bad"].sum() == df["label"].sum()
        assert result["approval_rate%"].iloc[-1] == pytest.approx(100.0)
        assert result["%bad"].sum() == pytest.approx(100.0)

    def test_no_missing_row_when_all_scores_in_range(self, score_df):
        result = describe_risk_score(score_df)
        assert "MISSING" not in set(result["band"])

    def test_no_bads_gives_zero_not_nan(self):
        df = pd.DataFrame({"score": [0.1, 0.6], "label": [0, 0]})
        result = describe_risk_score(df, segments=(0, 0.5, 1))
        assert not result[["%bad", "%bad_cumsum", "def_rate%"]].isna().any().any()
        assert (result["%bad"] == 0).all()

    def test_empty_leading_band_def_rate_is_zero(self):
        df = pd.DataFrame({"score": [0.9, 0.95], "label": [1, 0]})
        result = describe_risk_score(df, segments=(0, 0.5, 1))
        assert result["def_rate%"].iloc[0] == 0

    @pytest.mark.parametrize("labels", [[0, 2], [0, -1], [0, np.nan]])
    def test_invalid_labels_raise(self, labels):
        df = pd.DataFrame({"score": [0.1, 0.6], "label": labels})
        with pytest.raises(ValueError, match="only 0 and 1"):
            describe_risk_score(df)
