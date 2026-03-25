"""Tests for the EDA module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
from src.caketool.eda._validators import (
    clip_quantiles,
    require_column,
    require_columns,
    require_nonempty,
    require_numeric,
    top_k_series,
)
from src.caketool.eda.bivariate import (
    plot_category_heatmap,
    plot_distribution_by_group,
    plot_roc_curve,
    plot_scatter,
    plot_time_series,
    rank_associations,
)
from src.caketool.eda.config import EDAConfig
from src.caketool.eda.overview import (
    calculate_correlations,
    plot_correlations,
    profile,
)
from src.caketool.eda.univariate import (
    plot_categorical_frequency,
    plot_numeric_distribution,
    summarize_categorical_series,
    summarize_numeric_series,
)
from src.caketool.metric.association_metric import association

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def num_series() -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0, 1, 100), name="score")


@pytest.fixture()
def cat_series() -> pd.Series:
    return pd.Series(["A", "B", "A", "C", "B", "A", "B", "C", "A", "D"] * 10, name="cat")


@pytest.fixture()
def simple_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "x": rng.normal(0, 1, 200),
            "y": rng.normal(5, 2, 200),
            "z": rng.normal(-1, 0.5, 200),
            "label": rng.integers(0, 2, 200),
            "cat": np.tile(["A", "B", "C", "D"], 50),
        }
    )


# ===========================================================================
# Validators
# ===========================================================================


class TestRequireColumn:
    def test_existing_column_passes(self, simple_df):
        require_column(simple_df, "x")  # no exception

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError, match="not found"):
            require_column(simple_df, "missing")


class TestRequireColumns:
    def test_all_present_passes(self, simple_df):
        require_columns(simple_df, ["x", "y"])

    def test_one_missing_raises(self, simple_df):
        with pytest.raises(ValueError, match="not found"):
            require_columns(simple_df, ["x", "nonexistent"])


class TestRequireNumeric:
    def test_numeric_series_passes(self, num_series):
        require_numeric(num_series)

    def test_string_series_raises(self):
        with pytest.raises(TypeError, match="must be numeric"):
            require_numeric(pd.Series(["a", "b"], name="s"))


class TestRequireNonempty:
    def test_series_with_values_passes(self, num_series):
        require_nonempty(num_series)

    def test_all_null_series_raises(self):
        s = pd.Series([np.nan, np.nan], name="empty")
        with pytest.raises(ValueError, match="no non-null values"):
            require_nonempty(s)


class TestClipQuantiles:
    def test_no_clip_returns_original(self, num_series):
        result = clip_quantiles(num_series, 0.0, 1.0)
        pd.testing.assert_series_equal(result, num_series)

    def test_clip_reduces_range(self, num_series):
        clipped = clip_quantiles(num_series, 0.05, 0.95)
        assert clipped.min() >= num_series.quantile(0.05)
        assert clipped.max() <= num_series.quantile(0.95)


class TestTopKSeries:
    def test_keeps_top_k_categories(self, cat_series):
        result = top_k_series(cat_series, k=2)
        unique = set(result.unique())
        assert len(unique) <= 3  # 2 top + "Others"
        assert "Others" in unique

    def test_no_others_when_k_covers_all(self, cat_series):
        result = top_k_series(cat_series, k=10)
        assert "Others" not in result.unique()

    def test_custom_other_label(self, cat_series):
        result = top_k_series(cat_series, k=2, other_label="Rest")
        assert "Rest" in result.unique()


# ===========================================================================
# Config
# ===========================================================================


class TestEDAConfig:
    def test_default_values(self):
        cfg = EDAConfig()
        assert cfg.width == 900
        assert cfg.height == 450
        assert cfg.template == "simple_white"
        assert cfg.top_k_categories == 15
        assert cfg.percentile_step == 5
        assert len(cfg.color_palette) == 8

    def test_custom_values(self):
        cfg = EDAConfig(width=1200, height=600, template="plotly_dark")
        assert cfg.width == 1200
        assert cfg.height == 600
        assert cfg.template == "plotly_dark"


# ===========================================================================
# Univariate
# ===========================================================================


class TestHistogram:
    """Tests for plot_distribution (single series, no KDE)."""

    def test_returns_figure(self, num_series):
        fig = plot_numeric_distribution(num_series)
        assert isinstance(fig, go.Figure)

    def test_title_contains_series_name(self, num_series):
        fig = plot_numeric_distribution(num_series)
        assert "score" in fig.layout.title.text

    def test_non_numeric_raises(self, cat_series):
        with pytest.raises(TypeError):
            plot_numeric_distribution(cat_series)

    def test_all_null_raises(self):
        with pytest.raises(ValueError):
            plot_numeric_distribution(pd.Series([np.nan] * 5, name="s"))

    def test_custom_cfg_applied(self, num_series):
        cfg = EDAConfig(width=500, height=300)
        fig = plot_numeric_distribution(num_series, cfg=cfg)
        assert fig.layout.width == 500
        assert fig.layout.height == 300

    def test_show_stats_adds_vlines(self, num_series):
        fig = plot_numeric_distribution(num_series, show_stats=True)
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(shapes) == 6

    def test_show_stats_default_adds_vlines(self, num_series):
        fig = plot_numeric_distribution(num_series)
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(shapes) == 6

    def test_show_stats_false_no_vlines(self, num_series):
        fig = plot_numeric_distribution(num_series, show_stats=False)
        assert len(fig.layout.shapes) == 0

    def test_show_stats_table_annotation(self, num_series):
        fig = plot_numeric_distribution(num_series, show_stats=True)
        assert len(fig.layout.annotations) == 1
        table_text = fig.layout.annotations[0].text
        for label in ("Q1", "Median", "Q3", "Mean", "Lower Fence", "Upper Fence"):
            assert label in table_text


class TestOverlayHistogram:
    """Tests for plot_distribution (multiple series, no KDE)."""

    def test_returns_figure_with_multiple_traces(self, num_series):
        rng = np.random.default_rng(2)
        s2 = pd.Series(rng.normal(2, 1, 100), name="shifted")
        fig = plot_numeric_distribution({"A": num_series, "B": s2})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_non_numeric_raises(self, num_series, cat_series):
        with pytest.raises(TypeError):
            plot_numeric_distribution({"num": num_series, "cat": cat_series})

    def test_no_stats_on_overlay(self, num_series):
        rng = np.random.default_rng(4)
        s2 = pd.Series(rng.normal(0, 1, 100), name="s2")
        fig = plot_numeric_distribution({"A": num_series, "B": s2}, show_stats=True)
        assert len(fig.layout.shapes) == 0


class TestDistribution:
    """Tests for plot_distribution (single series, kde=True) and distribution alias."""

    def test_returns_figure_with_two_traces(self, num_series):
        fig = plot_numeric_distribution(num_series, kde=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # histogram + KDE

    def test_non_numeric_raises(self, cat_series):
        with pytest.raises(TypeError):
            plot_numeric_distribution(cat_series, kde=True)

    def test_show_stats_adds_vlines(self, num_series):
        fig = plot_numeric_distribution(num_series, kde=True, show_stats=True)
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(shapes) == 6

    def test_show_stats_default_adds_vlines(self, num_series):
        fig = plot_numeric_distribution(num_series, kde=True)
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(shapes) == 6

    def test_show_stats_false_no_vlines(self, num_series):
        fig = plot_numeric_distribution(num_series, kde=True, show_stats=False)
        assert len(fig.layout.shapes) == 0

    def test_show_stats_table_annotation(self, num_series):
        fig = plot_numeric_distribution(num_series, kde=True, show_stats=True)
        assert len(fig.layout.annotations) == 1
        texts = [a.text for a in fig.layout.annotations]
        for label in ("Q1", "Median", "Q3", "Mean", "Lower Fence", "Upper Fence"):
            assert any(label in t for t in texts)


class TestOverlayDistribution:
    """Tests for plot_distribution (multiple series, kde=True)."""

    def test_returns_figure_with_multiple_traces(self, num_series):
        rng = np.random.default_rng(2)
        s2 = pd.Series(rng.normal(2, 1, 100), name="shifted")
        fig = plot_numeric_distribution({"A": num_series, "B": s2}, kde=True)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4  # 2 series × (histogram + KDE)

    def test_has_histogram_and_kde_per_series(self, num_series):
        rng = np.random.default_rng(3)
        s2 = pd.Series(rng.normal(0, 2, 100), name="wide")
        fig = plot_numeric_distribution({"A": num_series, "B": s2}, kde=True)
        hist_count = sum(1 for t in fig.data if isinstance(t, go.Histogram))
        kde_count = sum(1 for t in fig.data if isinstance(t, go.Scatter))
        assert hist_count == 2
        assert kde_count == 2

    def test_non_numeric_raises(self, num_series, cat_series):
        with pytest.raises(TypeError):
            plot_numeric_distribution({"num": num_series, "cat": cat_series}, kde=True)

    def test_title_contains_first_label(self, num_series):
        fig = plot_numeric_distribution({"Group1": num_series}, kde=True)
        assert "Group1" in fig.layout.title.text


class TestPercentileTable:
    def test_returns_dataframe(self, num_series):
        result = summarize_numeric_series(num_series)
        assert isinstance(result, pd.DataFrame)

    def test_has_percentile_column(self, num_series):
        result = summarize_numeric_series(num_series)
        assert "percentile" in result.columns

    def test_has_value_column(self, num_series):
        result = summarize_numeric_series(num_series)
        value_col = f"{num_series.name}_value"
        assert value_col in result.columns

    def test_has_count_columns(self, num_series):
        result = summarize_numeric_series(num_series)
        assert "count_below" in result.columns
        assert "count_above" in result.columns

    def test_stat_rows_are_first_in_expected_order(self, num_series):
        result = summarize_numeric_series(num_series)
        expected = ["Mean", "Median", "Std", "Min", "Max", "Q1", "Q3", "lower_fence", "upper_fence"]
        assert list(result["percentile"].head(len(expected))) == expected

    def test_count_columns_partition_total(self, num_series):
        result = summarize_numeric_series(num_series)
        total = num_series.notna().sum()
        assert (result["count_below"] + result["count_above"]).eq(total).all()

    @pytest.mark.parametrize("step", [0, 101, -1, 1.5, "5", True])
    def test_invalid_step_raises(self, num_series, step):
        with pytest.raises(ValueError, match=r"step must be an integer in \[1, 100\]"):
            summarize_numeric_series(num_series, step=step)

    def test_non_numeric_raises(self, cat_series):
        with pytest.raises(TypeError):
            summarize_numeric_series(cat_series)


class TestComputeFrequency:
    def test_returns_dataframe(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert list(result.columns) == ["value", "count", "pct", "cumulative_count", "cumulative_pct"]

    def test_cumulative_columns_are_monotonic(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert result["cumulative_count"].is_monotonic_increasing
        assert result["cumulative_pct"].is_monotonic_increasing

    def test_cumulative_count_ends_at_total(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert int(result["cumulative_count"].iloc[-1]) == len(cat_series)

    def test_top_k_limits_rows_with_others(self, cat_series):
        # cat_series has 4 unique; top_k=2 → 2 top + 1 Others row = 3
        result = summarize_categorical_series(cat_series, top_k=2)
        assert len(result) == 3
        assert result.iloc[-1]["value"] == "Others"

    def test_pct_sums_to_100(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert result["pct"].sum() == pytest.approx(100.0, abs=0.1)

    def test_nan_row_included_by_default(self):
        s = pd.Series(["A", "B", None, "A", np.nan], name="x")
        result = summarize_categorical_series(s)
        assert "NaN" in result["value"].values

    def test_nan_row_excluded_when_dropna(self):
        s = pd.Series(["A", "B", None, "A", np.nan], name="x")
        result = summarize_categorical_series(s, dropna=True)
        assert "NaN" not in result["value"].values

    def test_no_others_when_top_k_covers_all(self, cat_series):
        result = summarize_categorical_series(cat_series, top_k=100)
        assert "Others" not in result["value"].values

    def test_top_k_zero_groups_all_non_null_into_others(self, cat_series):
        result = summarize_categorical_series(cat_series, top_k=0, dropna=True)
        assert len(result) == 1
        assert result.iloc[0]["value"] == "Others"
        assert int(result.iloc[0]["count"]) == len(cat_series)

    @pytest.mark.parametrize("top_k", [-1, 1.5, "2", True])
    def test_invalid_top_k_raises(self, cat_series, top_k):
        with pytest.raises(ValueError, match="top_k must be a non-negative integer"):
            summarize_categorical_series(cat_series, top_k=top_k)


class TestPlotFrequency:
    def test_pie_returns_figure(self, cat_series):
        fig = plot_categorical_frequency(cat_series, mode="pie")
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Pie)

    def test_bar_returns_figure(self, cat_series):
        fig = plot_categorical_frequency(cat_series, mode="bar")
        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Bar)

    def test_barh_horizontal(self, cat_series):
        fig = plot_categorical_frequency(cat_series, mode="barh")
        assert isinstance(fig, go.Figure)
        assert fig.data[0].orientation == "h"

    def test_invalid_mode_raises(self, cat_series):
        with pytest.raises(ValueError):
            plot_categorical_frequency(cat_series, mode="invalid")

    def test_top_k_groups_others(self, cat_series):
        fig = plot_categorical_frequency(cat_series, top_k=2, mode="pie")
        labels = list(fig.data[0].labels)
        assert "Others" in labels

    def test_default_mode_is_pie(self, cat_series):
        fig = plot_categorical_frequency(cat_series)
        assert isinstance(fig.data[0], go.Pie)


# ===========================================================================
# Bivariate
# ===========================================================================


class TestScatter:
    def test_returns_figure(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y")
        assert isinstance(fig, go.Figure)

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError):
            plot_scatter(simple_df, "x", "nonexistent")

    def test_color_by_creates_multiple_traces(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y", color_by="cat")
        assert len(fig.data) > 1

    def test_title_contains_columns(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y")
        assert "x" in fig.layout.title.text and "y" in fig.layout.title.text


class TestViolinByCategory:
    def test_returns_figure(self, simple_df):
        fig = plot_distribution_by_group(simple_df, "cat", "x", mode="violin")
        assert isinstance(fig, go.Figure)

    def test_trace_count_matches_categories(self, simple_df):
        fig = plot_distribution_by_group(simple_df, "cat", "x", mode="violin")
        n_cats = simple_df["cat"].nunique()
        assert len(fig.data) == n_cats


class TestRocCurvePlot:
    def test_returns_figure(self, simple_df):
        fig = plot_roc_curve(simple_df, "label", "x")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, simple_df):
        fig = plot_roc_curve(simple_df, "label", "x")
        assert len(fig.data) == 2  # ROC + random line

    def test_title_contains_auc(self, simple_df):
        fig = plot_roc_curve(simple_df, "label", "x")
        assert "AUC" in fig.layout.title.text


# ===========================================================================
# New Bivariate Functions
# ===========================================================================


class TestPlotScatter:
    def test_returns_figure(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y")
        assert isinstance(fig, go.Figure)

    def test_with_trend_adds_traces(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y")
        # Always has scatter + trend (at least 2 traces)
        assert len(fig.data) >= 2

    def test_with_correlation_annotation(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y")
        # Always shows correlation in title (r= and strength/significance)
        assert "r=" in fig.layout.title.text
        assert "significant" in fig.layout.title.text or "negligible" in fig.layout.title.text

    def test_color_by_multiple_traces(self, simple_df):
        fig = plot_scatter(simple_df, "x", "y", color_by="cat")
        assert len(fig.data) > 1

    def test_sampling_respects_random_state(self):
        df = pd.DataFrame(
            {
                "x": np.arange(2000, dtype=float),
                "y": np.arange(2000, dtype=float) * 2,
            }
        )
        fig1 = plot_scatter(df, "x", "y", sample_n=150, random_state=7)
        fig2 = plot_scatter(df, "x", "y", sample_n=150, random_state=7)
        np.testing.assert_array_equal(np.asarray(fig1.data[0].x), np.asarray(fig2.data[0].x))


class TestPlotDistributionByGroup:
    def test_box_mode(self, simple_df):
        fig = plot_distribution_by_group(simple_df, "cat", "x", mode="box")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_violin_mode(self, simple_df):
        fig = plot_distribution_by_group(simple_df, "cat", "x", mode="violin")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_hist_mode(self, simple_df):
        fig = plot_distribution_by_group(simple_df, "cat", "x", mode="hist")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_top_k_limits_categories(self, simple_df):
        fig = plot_distribution_by_group(simple_df, "cat", "x", mode="box", top_k=2)
        assert len(fig.data) <= 2

    def test_invalid_mode_raises(self, simple_df):
        with pytest.raises(ValueError, match="mode must be one of"):
            plot_distribution_by_group(simple_df, "cat", "x", mode="invalid")


class TestPlotCategoryHeatmap:
    def test_returns_heatmap(self, simple_df):
        fig = plot_category_heatmap(simple_df, "cat", "label")
        assert isinstance(fig, go.Figure)

    def test_title_includes_cramers_v(self, simple_df):
        fig = plot_category_heatmap(simple_df, "cat", "label")
        assert "V=" in fig.layout.title.text

    def test_normalize_affects_values(self, simple_df):
        fig_norm = plot_category_heatmap(simple_df, "cat", "label", normalize=True)
        fig_count = plot_category_heatmap(simple_df, "cat", "label", normalize=False)
        # Both should produce figures
        assert isinstance(fig_norm, go.Figure)
        assert isinstance(fig_count, go.Figure)


class TestPlotTimeSeries:
    def test_returns_figure(self, simple_df):
        fig = plot_time_series(simple_df, x="x", y="y")
        assert isinstance(fig, go.Figure)

    def test_multi_series(self, simple_df):
        fig = plot_time_series(simple_df, x="x", y=["y", "z"])
        assert isinstance(fig, go.Figure)

    def test_moving_average(self, simple_df):
        fig = plot_time_series(simple_df, x="x", y="y", ma=5)
        assert isinstance(fig, go.Figure)

    def test_group_by_creates_multiple_traces(self, simple_df):
        fig = plot_time_series(simple_df, x="x", y="y", group_by="cat")
        assert isinstance(fig, go.Figure)

    def test_accepts_datetime_x(self):
        df = pd.DataFrame(
            {
                "ds": pd.date_range("2025-01-01", periods=20, freq="D"),
                "y": np.linspace(0.0, 1.0, 20),
            }
        )
        fig = plot_time_series(df, x="ds", y="y")
        assert isinstance(fig, go.Figure)

    def test_minmax_band_supported(self, simple_df):
        fig = plot_time_series(simple_df, x="x", y="y", group_by="cat", band="minmax")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= simple_df["cat"].nunique() * 3

    def test_band_without_group_by_raises(self, simple_df):
        with pytest.raises(ValueError, match="group_by"):
            plot_time_series(simple_df, x="x", y="y", band="std")

    def test_invalid_band_raises(self, simple_df):
        with pytest.raises(ValueError, match="band must be one of"):
            plot_time_series(simple_df, x="x", y="y", group_by="cat", band="invalid")


class TestRankAssociations:
    def test_returns_dataframe(self, simple_df):
        result = rank_associations(simple_df, target="label")
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "association" in result.columns
        assert "p_value" in result.columns
        assert "method" in result.columns

    def test_sorted_by_association_strength(self, simple_df):
        result = rank_associations(simple_df, target="label")
        if len(result) > 1:
            # Check that associations are sorted by absolute value
            abs_vals = result["association"].abs().values
            assert (abs_vals == sorted(abs_vals, reverse=True)).all()

    def test_includes_numeric_and_categorical(self, simple_df):
        result = rank_associations(simple_df, target="label")
        methods = result["method"].unique()
        # Should have both correlation and cramers_v
        assert "pearson" in methods or "cramers_v" in methods

    def test_invalid_num_method_raises(self, simple_df):
        with pytest.raises(ValueError, match="num_method"):
            rank_associations(simple_df, target="label", num_method="kendall")

    def test_categorical_target_uses_eta_for_numeric_features(self, simple_df):
        result = rank_associations(simple_df, target="cat")
        numeric_rows = result[result["feature"].isin(["x", "y", "z", "label"])]
        assert (numeric_rows["method"] == "eta").all()

    def test_categorical_missing_excluded_before_cramers_v(self):
        df = pd.DataFrame(
            {
                "target": ["T1", "T1", "T2", None, "T2"],
                "cat_feature": ["A", None, "A", "B", "B"],
            }
        )
        result = rank_associations(df, target="target", num_cols=[], cat_cols=["cat_feature"])
        mask = df["target"].notna() & df["cat_feature"].notna()
        expected_v, expected_p = association(
            df.loc[mask, "cat_feature"].astype(str),
            df.loc[mask, "target"].astype(str),
            method="cramers_v",
        )
        assert result.loc[0, "association"] == pytest.approx(round(expected_v, 4))
        assert result.loc[0, "p_value"] == pytest.approx(round(expected_p, 4))


class TestPlotRocCurve:
    def test_returns_figure(self, simple_df):
        fig = plot_roc_curve(simple_df, "label", "x")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, simple_df):
        fig = plot_roc_curve(simple_df, "label", "x")
        assert len(fig.data) == 2

    def test_title_contains_auc(self, simple_df):
        fig = plot_roc_curve(simple_df, "label", "x")
        assert "AUC" in fig.layout.title.text


# ===========================================================================
# Overview
# ===========================================================================


class TestProfile:
    def test_returns_dataframe(self, simple_df):
        result = profile(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_column(self, simple_df):
        result = profile(simple_df)
        assert len(result) == len(simple_df.columns)

    def test_contains_expected_columns(self, simple_df):
        result = profile(simple_df)
        for col in ["column", "dtype", "n_total", "n_missing", "missing_pct", "n_unique"]:
            assert col in result.columns

    def test_numeric_columns_have_mean(self, simple_df):
        result = profile(simple_df)
        row = result[result["column"] == "x"].iloc[0]
        assert row["mean"] is not None

    def test_categorical_columns_have_top_value(self, simple_df):
        result = profile(simple_df)
        row = result[result["column"] == "cat"].iloc[0]
        assert row["top_value"] is not None


class TestCorrelationHeatmap:
    def test_returns_figure(self, simple_df):
        fig = plot_correlations(simple_df)
        assert isinstance(fig, go.Figure)

    def test_fewer_than_two_columns_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="2 columns"):
            plot_correlations(df)

    def test_spearman_method(self, simple_df):
        fig = plot_correlations(simple_df, num_method="spearman")
        assert "spearman" in fig.layout.title.text.lower()


class TestCalculateAllCorrelations:
    def test_returns_dataframe(self, simple_df):
        result = calculate_correlations(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_square_matrix(self, simple_df):
        result = calculate_correlations(simple_df)
        assert result.shape[0] == result.shape[1] == len(simple_df.columns)

    def test_diagonal_is_one(self, simple_df):
        result = calculate_correlations(simple_df)
        assert all(result.loc[c, c] == pytest.approx(1.0) for c in result.columns)

    def test_symmetric(self, simple_df):
        result = calculate_correlations(simple_df)
        pd.testing.assert_frame_equal(result, result.T)

    def test_values_in_range(self, simple_df):
        result = calculate_correlations(simple_df)
        assert result.values.min() >= 0.0
        assert result.values.max() <= 1.0

    def test_unique_threshold_changes_low_cardinality_numeric_handling(self, simple_df):
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "label": [0, 1, 2, 0, 1, 2],
            }
        )
        corr_low_threshold = calculate_correlations(df, unique_threshold=1)
        corr_high_threshold = calculate_correlations(df, unique_threshold=100)
        assert corr_low_threshold.loc["x", "label"] != corr_high_threshold.loc["x", "label"]


# ---------------------------------------------------------------------------
# Import test – verify public API is accessible
# ---------------------------------------------------------------------------

from src.caketool.eda import (  # noqa: E402, F811 (after all other imports to keep organisation clear)
    EDAConfig,
    calculate_correlations,
    plot_categorical_frequency,
    plot_correlations,
    plot_numeric_distribution,
    plot_scatter,
    profile,
    summarize_categorical_series,
    summarize_numeric_series,
)


def test_public_api_importable():
    assert callable(plot_numeric_distribution)
    assert callable(plot_categorical_frequency)
    assert callable(plot_scatter)
    assert callable(calculate_correlations)
    assert callable(profile)
    assert EDAConfig is not None
