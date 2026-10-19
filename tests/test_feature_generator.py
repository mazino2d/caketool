import pandas as pd
import pytest
from src.caketool.feature import generate_features_by_window


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "report_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-02", "2024-01-08"]),
            "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 5),
            "amount": [100.0, 200.0, 150.0, 50.0, 75.0],
            "category": ["A", "B", "A", "A", "B"],
            "is_active": [True, False, True, True, False],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-02", "2024-01-08"]),
        }
    )


class TestGenerateFeaturesBasic:
    """Test basic functionality of generate_features_by_window."""

    def test_numeric_cols_lifetime(self, sample_df):
        """Test numeric column aggregations with lifetime window."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            numeric_cols=("amount",),
            backend="pandas",
        )

        assert "user_id" in result.columns
        assert "fs_event_timestamp" in result.columns
        assert "ft_all_amount_lifetime_min" in result.columns
        assert "ft_all_amount_lifetime_max" in result.columns
        assert "ft_all_amount_lifetime_avg" in result.columns
        assert "ft_all_amount_lifetime_sum" in result.columns
        assert "ft_all_amount_lifetime_cnt" in result.columns
        assert len(result) == 2  # 2 unique users

    def test_string_cols_lifetime(self, sample_df):
        """Test string column aggregations."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            string_cols=("category",),
            backend="pandas",
        )

        assert "ft_all_category_lifetime_cnt" in result.columns
        assert "ft_all_category_lifetime_nunique" in result.columns
        assert "ft_all_category_lifetime_unique_ratio" in result.columns
        assert "ft_all_category_lifetime_entropy" not in result.columns

    def test_boolean_cols_lifetime(self, sample_df):
        """Test boolean column aggregations."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            numeric_cols=("amount",),
            boolean_cols=("is_active",),
            backend="pandas",
        )

        assert "ft_all_is_active_lifetime_poscnt" in result.columns
        assert "ft_all_is_active_lifetime_posratio" in result.columns

    def test_date_cols_lifetime(self, sample_df):
        """Test date column aggregations."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            date_cols=("event_date",),
            backend="pandas",
        )

        assert "ft_all_event_date_lifetime_firstdatediff" in result.columns
        assert "ft_all_event_date_lifetime_lastdatediff" in result.columns
        assert "ft_all_event_date_lifetime_daysbetween" in result.columns


class TestGenerateFeaturesLookback:
    """Test lookback window functionality."""

    def test_multiple_lookback_days(self, sample_df):
        """Test with multiple lookback windows."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            lookback_days=(0, 7),
            numeric_cols=("amount",),
            backend="pandas",
        )

        # Check lifetime features
        assert "ft_all_amount_lifetime_sum" in result.columns
        # Check 7-day features
        assert "ft_all_amount_d7_sum" in result.columns

    def test_lookback_filters_data(self, sample_df):
        """Test that lookback window filters data correctly."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            lookback_days=(7,),
            numeric_cols=("amount",),
            backend="pandas",
        )

        # Only events within 7 days of fs_event_timestamp should be included
        assert "ft_all_amount_d7_sum" in result.columns


class TestGenerateFeaturesKeyCol:
    """Test key column functionality."""

    def test_specific_key_col(self, sample_df):
        """Test with specific key column instead of __all__."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            key_cols=("category",),
            numeric_cols=("amount",),
            backend="pandas",
        )

        # Should have features pivoted by category values (a, b)
        assert "user_id" in result.columns
        # Features should be pivoted by category
        feature_cols = [c for c in result.columns if c.startswith("ft_")]
        assert len(feature_cols) > 0

    def test_custom_feature_prefix(self, sample_df):
        """Test custom feature prefix."""
        result = generate_features_by_window(
            sample_df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            numeric_cols=("amount",),
            feature_prefix="custom",
            backend="pandas",
        )

        feature_cols = [c for c in result.columns if c.startswith("custom_")]
        assert len(feature_cols) > 0


class TestGenerateFeaturesErrors:
    """Test error handling."""

    def test_no_columns_provided(self, sample_df):
        """Test error when no feature columns provided."""
        with pytest.raises(ValueError, match="At least one of"):
            generate_features_by_window(
                sample_df,
                client_id_col="user_id",
                report_date_col="report_date",
                fs_event_timestamp="fs_event_timestamp",
                backend="pandas",
            )

    def test_unsupported_backend(self, sample_df):
        """Test error for unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            generate_features_by_window(
                sample_df,
                client_id_col="user_id",
                report_date_col="report_date",
                fs_event_timestamp="fs_event_timestamp",
                numeric_cols=("amount",),
                backend="invalid_backend",
            )

    def test_negative_lookback_days(self, sample_df):
        """Test error for negative lookback days."""
        with pytest.raises(ValueError, match="positive integer or zero"):
            generate_features_by_window(
                sample_df,
                client_id_col="user_id",
                report_date_col="report_date",
                fs_event_timestamp="fs_event_timestamp",
                lookback_days=(-7,),
                numeric_cols=("amount",),
                backend="pandas",
            )


class TestGenerateFeaturesValues:
    """Test correct calculation of feature values."""

    def test_numeric_aggregations_values(self):
        """Test that numeric aggregations are calculated correctly."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "report_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 3),
                "amount": [10.0, 20.0, 30.0],
            }
        )

        result = generate_features_by_window(
            df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            numeric_cols=("amount",),
            backend="pandas",
        )

        row = result[result["user_id"] == "u1"].iloc[0]
        assert row["ft_all_amount_lifetime_min"] == 10.0
        assert row["ft_all_amount_lifetime_max"] == 30.0
        assert row["ft_all_amount_lifetime_sum"] == 60.0
        assert row["ft_all_amount_lifetime_cnt"] == 3
        assert row["ft_all_amount_lifetime_avg"] == 20.0

    def test_boolean_aggregations_values(self):
        """Test that boolean aggregations are calculated correctly."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u1"],
                "report_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
                "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 4),
                "amount": [10.0, 20.0, 30.0, 40.0],
                "is_active": [True, True, False, True],
            }
        )

        result = generate_features_by_window(
            df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            numeric_cols=("amount",),
            boolean_cols=("is_active",),
            backend="pandas",
        )

        row = result[result["user_id"] == "u1"].iloc[0]
        assert row["ft_all_is_active_lifetime_poscnt"] == 3
        assert row["ft_all_is_active_lifetime_posratio"] == 0.75

    def test_string_nunique_values(self):
        """Test that nunique is calculated correctly."""
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u1"],
                "report_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
                "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 4),
                "category": ["A", "B", "A", "C"],
            }
        )

        result = generate_features_by_window(
            df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            string_cols=("category",),
            backend="pandas",
        )

        row = result[result["user_id"] == "u1"].iloc[0]
        assert row["ft_all_category_lifetime_cnt"] == 4
        assert row["ft_all_category_lifetime_nunique"] == 3


class TestGenerateFeaturesPolars:
    """Test polars backend."""

    def test_polars_backend(self):
        """Test basic functionality with polars backend."""
        pytest.importorskip("polars")
        import polars as pl

        df = pl.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "report_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "fs_event_timestamp": ["2024-01-15"] * 3,
                "amount": [10.0, 20.0, 30.0],
            }
        ).with_columns(
            [
                pl.col("report_date").str.to_datetime(),
                pl.col("fs_event_timestamp").str.to_datetime(),
            ]
        )

        result = generate_features_by_window(
            df,
            client_id_col="user_id",
            report_date_col="report_date",
            fs_event_timestamp="fs_event_timestamp",
            numeric_cols=("amount",),
            backend="polars",
        )

        assert "user_id" in result.columns
        assert "ft_all_amount_lifetime_sum" in result.columns


def _event_df():
    """Events for two users with nulls, a numeric-like key and look-alike key values."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u2", "u2"],
            "report_date": pd.to_datetime(
                ["2024-01-01", "2024-01-09", "2024-01-12", "2024-01-14", "2024-01-03", "2024-01-10", "2024-01-11"]
            ),
            "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 7),
            "amount": [10.0, 20.0, 35.0, 40.0, 5.0, None, 15.0],
            "product": ["card", "credit_card", "card", None, "Card", "credit_card", "credit_card"],
            "merchant": ["m1", "m2", None, "m1", "m3", "m3", None],
            "is_active": pd.array([True, None, False, True, True, True, None], dtype="boolean"),
            "event_date": pd.to_datetime(
                ["2024-01-01", "2024-01-09", "2024-01-12", "2024-01-14", "2024-01-03", "2024-01-10", "2024-01-11"]
            ),
        }
    )


class TestGenerateFeaturesLeakage:
    """Lifetime features must only use history before the snapshot."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_future_rows_excluded_from_lifetime(self, backend):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "report_date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-03-01"]),
                "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 3),
                "amount": [10.0, 1000.0, 5000.0],
            }
        )
        if backend == "polars":
            pl = pytest.importorskip("polars")
            df = pl.from_pandas(df)

        result = generate_features_by_window(df, numeric_cols=("amount",), backend=backend)
        if backend == "polars":
            result = result.to_pandas()

        row = result.iloc[0]
        # Rows on or after fs_event_timestamp are excluded, same as windowed features
        assert row["ft_all_amount_lifetime_cnt"] == 1
        assert row["ft_all_amount_lifetime_max"] == 10.0


class TestGenerateFeaturesSchema:
    """The output schema must not depend on the data."""

    def test_std_kept_when_every_group_has_one_row(self):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u2"],
                "report_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 2),
                "amount": [10.0, 20.0],
            }
        )

        result = generate_features_by_window(df, numeric_cols=("amount",), backend="pandas")

        assert "ft_all_amount_lifetime_std" in result.columns
        assert result["ft_all_amount_lifetime_std"].isna().all()
        assert len(result) == 2


class TestGenerateFeaturesKeyValues:
    """Key column normalisation and pivot naming."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_lookalike_key_values_not_swapped(self, backend):
        df = _event_df()
        if backend == "polars":
            pl = pytest.importorskip("polars")
            df = pl.from_pandas(df)

        result = generate_features_by_window(df, key_cols=("product",), numeric_cols=("amount",), backend=backend)
        if backend == "polars":
            result = result.to_pandas()
        u1 = result[result["user_id"] == "u1"].iloc[0]

        assert u1["ft_card_amount_lifetime_sum"] == 45.0
        assert u1["ft_credit_card_amount_lifetime_sum"] == 20.0
        # Null keys are pivoted under the "null" label
        assert u1["ft_null_amount_lifetime_sum"] == 40.0

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_numeric_key_col(self, backend):
        df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1"],
                "report_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "fs_event_timestamp": pd.to_datetime(["2024-01-15"] * 3),
                "channel": [1, 2, 1],
                "amount": [10.0, 20.0, 30.0],
            }
        )
        if backend == "polars":
            pl = pytest.importorskip("polars")
            df = pl.from_pandas(df)

        result = generate_features_by_window(df, key_cols=("channel",), numeric_cols=("amount",), backend=backend)
        if backend == "polars":
            result = result.to_pandas()

        assert result.iloc[0]["ft_1_amount_lifetime_sum"] == 40.0
        assert result.iloc[0]["ft_2_amount_lifetime_sum"] == 20.0


class TestGenerateFeaturesNulls:
    """Null values are skipped, not counted as a value."""

    @pytest.mark.parametrize("backend", ["pandas", "polars"])
    def test_null_handling(self, backend):
        df = _event_df()
        if backend == "polars":
            pl = pytest.importorskip("polars")
            df = pl.from_pandas(df)

        result = generate_features_by_window(
            df, string_cols=("merchant",), boolean_cols=("is_active",), backend=backend
        )
        if backend == "polars":
            result = result.to_pandas()
        u1 = result[result["user_id"] == "u1"].iloc[0]

        assert u1["ft_all_merchant_lifetime_cnt"] == 3
        assert u1["ft_all_merchant_lifetime_nunique"] == 2
        assert u1["ft_all_merchant_lifetime_unique_ratio"] == pytest.approx(2 / 3)
        assert u1["ft_all_is_active_lifetime_poscnt"] == 2
        assert u1["ft_all_is_active_lifetime_posratio"] == pytest.approx(2 / 3)


class TestGenerateFeaturesBackendConsistency:
    """pandas and polars must produce the same features for the same input."""

    def test_pandas_matches_polars(self):
        pl = pytest.importorskip("polars")
        df = _event_df()
        kwargs = dict(
            key_cols=("__all__", "product"),
            lookback_days=(0, 7),
            numeric_cols=("amount",),
            string_cols=("merchant",),
            date_cols=("event_date",),
            boolean_cols=("is_active",),
        )

        pd_result = generate_features_by_window(df, backend="pandas", **kwargs)
        pl_result = generate_features_by_window(pl.from_pandas(df), backend="polars", **kwargs).to_pandas()

        assert sorted(pd_result.columns) == sorted(pl_result.columns)
        pd_result = pd_result.sort_values("user_id").reset_index(drop=True)
        pl_result = pl_result.sort_values("user_id").reset_index(drop=True)[pd_result.columns]
        feature_cols = [c for c in pd_result.columns if c.startswith("ft_")]
        pd.testing.assert_frame_equal(
            pd_result[feature_cols].astype("float64"),
            pl_result[feature_cols].astype("float64"),
            check_exact=False,
        )
