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
    bar_category_vs_category,
    box_by_category,
    correlation_table,
    cramers_v,
    cramers_v_target,
    histogram_by_label,
    line_with_ma,
    roc_curve_plot,
    scatter,
    violin_by_category,
)
from src.caketool.eda.config import EDAConfig
from src.caketool.eda.multivariate import (
    parallel_coordinates,
    scatter_3d,
    scatter_matrix,
    stacked_bar,
)
from src.caketool.eda.overview import (
    correlation_heatmap,
    cramers_v_heatmap,
    pivot_count,
    pivot_rate,
    profile,
    top_extreme_values,
)
from src.caketool.eda.quality import (
    duplicate_columns,
    duplicate_rows,
    missing_summary,
    psi,
    psi_category,
    psi_report,
)
from src.caketool.eda.univariate import (
    plot_categorical_frequency,
    plot_numeric_distribution,
    summarize_categorical_series,
    summarize_numeric_series,
)

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

    def test_non_numeric_raises(self, cat_series):
        with pytest.raises(TypeError):
            summarize_numeric_series(cat_series)


class TestComputeFrequency:
    def test_returns_dataframe(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, cat_series):
        result = summarize_categorical_series(cat_series)
        assert list(result.columns) == ["value", "count", "pct"]

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
        fig = scatter(simple_df, "x", "y")
        assert isinstance(fig, go.Figure)

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError):
            scatter(simple_df, "x", "nonexistent")

    def test_color_by_creates_multiple_traces(self, simple_df):
        fig = scatter(simple_df, "x", "y", color_by="cat")
        assert len(fig.data) > 1

    def test_title_contains_columns(self, simple_df):
        fig = scatter(simple_df, "x", "y")
        assert "x" in fig.layout.title.text and "y" in fig.layout.title.text


class TestLineWithMa:
    def test_returns_figure_with_ma(self, simple_df):
        fig = line_with_ma(simple_df, "x", "y", ma=5)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # raw line + MA

    def test_no_ma_returns_single_trace(self, simple_df):
        fig = line_with_ma(simple_df, "x", "y", ma=0)
        assert len(fig.data) == 1


class TestBoxByCategory:
    def test_returns_figure(self, simple_df):
        fig = box_by_category(simple_df, "cat", "x")
        assert isinstance(fig, go.Figure)

    def test_non_numeric_num_col_raises(self, simple_df):
        with pytest.raises(TypeError):
            box_by_category(simple_df, "x", "cat")

    def test_top_k_limits_traces(self, simple_df):
        fig = box_by_category(simple_df, "cat", "x", top_k=2)
        assert len(fig.data) == 2


class TestViolinByCategory:
    def test_returns_figure(self, simple_df):
        fig = violin_by_category(simple_df, "cat", "x")
        assert isinstance(fig, go.Figure)

    def test_trace_count_matches_categories(self, simple_df):
        fig = violin_by_category(simple_df, "cat", "x")
        n_cats = simple_df["cat"].nunique()
        assert len(fig.data) == n_cats


class TestHistogramByLabel:
    def test_returns_figure(self, simple_df):
        fig = histogram_by_label(simple_df, "x", "label")
        assert isinstance(fig, go.Figure)

    def test_traces_equal_unique_labels(self, simple_df):
        fig = histogram_by_label(simple_df, "x", "label")
        n_labels = simple_df["label"].nunique()
        assert len(fig.data) == n_labels


class TestBarCategoryVsCategory:
    def test_returns_figure(self, simple_df):
        fig = bar_category_vs_category(simple_df, "cat", "label")
        assert isinstance(fig, go.Figure)

    def test_normalize_changes_y_title(self, simple_df):
        fig = bar_category_vs_category(simple_df, "cat", "label", normalize=True)
        assert "%" in fig.layout.yaxis.title.text


class TestCramersV:
    def test_identical_series_returns_one(self):
        s = pd.Series(["A", "B", "A", "C"] * 25)
        result = cramers_v(s, s)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_independent_series_returns_near_zero(self):
        rng = np.random.default_rng(42)
        s1 = pd.Series(rng.choice(["X", "Y"], 500))
        s2 = pd.Series(rng.choice(["P", "Q"], 500))
        result = cramers_v(s1, s2)
        assert result < 0.1

    def test_returns_float_in_range(self):
        s1 = pd.Series(["A", "B"] * 50)
        s2 = pd.Series(["A", "B"] * 50)
        result = cramers_v(s1, s2)
        assert 0.0 <= result <= 1.0


class TestCramersVTarget:
    def test_returns_dataframe_with_correct_columns(self, simple_df):
        result = cramers_v_target(simple_df, target="label")
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "cramers_v" in result.columns

    def test_sorted_descending(self, simple_df):
        result = cramers_v_target(simple_df, target="label")
        assert result["cramers_v"].is_monotonic_decreasing

    def test_missing_target_raises(self, simple_df):
        with pytest.raises(ValueError):
            cramers_v_target(simple_df, target="nonexistent")


class TestCorrelationTable:
    def test_returns_dataframe(self, simple_df):
        result = correlation_table(simple_df)
        assert isinstance(result, pd.DataFrame)
        assert "col1" in result.columns
        assert "col2" in result.columns
        assert "correlation" in result.columns

    def test_threshold_filters_rows(self):
        # Use a DataFrame with known high correlation to ensure filtering works
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        df = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(0, 0.01, 200), "c": rng.normal(0, 1, 200)})
        result_all = correlation_table(df, threshold=0.0)
        result_filtered = correlation_table(df, threshold=0.9)
        assert len(result_filtered) < len(result_all)

    def test_spearman_method(self, simple_df):
        result = correlation_table(simple_df, method="spearman")
        assert isinstance(result, pd.DataFrame)


class TestRocCurvePlot:
    def test_returns_figure(self, simple_df):
        fig = roc_curve_plot(simple_df, "label", "x")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self, simple_df):
        fig = roc_curve_plot(simple_df, "label", "x")
        assert len(fig.data) == 2  # ROC + random line

    def test_title_contains_auc(self, simple_df):
        fig = roc_curve_plot(simple_df, "label", "x")
        assert "AUC" in fig.layout.title.text


# ===========================================================================
# Multivariate
# ===========================================================================


class TestParallelCoordinates:
    def test_returns_figure(self, simple_df):
        fig = parallel_coordinates(simple_df, dims=["x", "y", "z"])
        assert isinstance(fig, go.Figure)

    def test_missing_dim_raises(self, simple_df):
        with pytest.raises(ValueError):
            parallel_coordinates(simple_df, dims=["x", "nonexistent"])


class TestScatter3d:
    def test_returns_figure(self, simple_df):
        fig = scatter_3d(simple_df, "x", "y", "z")
        assert isinstance(fig, go.Figure)

    def test_color_by_creates_multiple_traces(self, simple_df):
        fig = scatter_3d(simple_df, "x", "y", "z", color_by="cat")
        assert len(fig.data) > 1


class TestScatterMatrix:
    def test_returns_figure(self, simple_df):
        fig = scatter_matrix(simple_df, columns=["x", "y", "z"])
        assert isinstance(fig, go.Figure)

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError):
            scatter_matrix(simple_df, columns=["x", "nonexistent"])


class TestStackedBar:
    def test_returns_figure(self, simple_df):
        fig = stacked_bar(simple_df, x="cat", category="label")
        assert isinstance(fig, go.Figure)

    def test_normalize_changes_y_title(self, simple_df):
        fig = stacked_bar(simple_df, x="cat", category="label", normalize=True)
        assert "%" in fig.layout.yaxis.title.text

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError):
            stacked_bar(simple_df, x="nonexistent", category="label")


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
        fig = correlation_heatmap(simple_df)
        assert isinstance(fig, go.Figure)

    def test_fewer_than_two_numeric_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="2 numeric"):
            correlation_heatmap(df)

    def test_spearman_method(self, simple_df):
        fig = correlation_heatmap(simple_df, method="spearman")
        assert "spearman" in fig.layout.title.text.lower()


class TestCramersVHeatmap:
    def test_returns_figure(self, simple_df):
        fig = cramers_v_heatmap(simple_df, cat_cols=["cat", "label"])
        assert isinstance(fig, go.Figure)

    def test_fewer_than_two_columns_raises(self, simple_df):
        with pytest.raises(ValueError, match="2 categorical"):
            cramers_v_heatmap(simple_df, cat_cols=["cat"])


class TestPivotCount:
    def test_returns_dataframe(self, simple_df):
        result = pivot_count(simple_df, index="cat", columns="label")
        assert isinstance(result, pd.DataFrame)

    def test_contains_total_column_when_margins(self, simple_df):
        result = pivot_count(simple_df, index="cat", columns="label", margins=True)
        assert "Total" in result.columns

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError):
            pivot_count(simple_df, index="cat", columns="nonexistent")


class TestPivotRate:
    def test_returns_dataframe(self, simple_df):
        result = pivot_rate(simple_df, index="cat", columns="label", target="x")
        assert isinstance(result, pd.DataFrame)

    def test_invalid_aggfunc_raises(self, simple_df):
        with pytest.raises(ValueError, match="aggfunc"):
            pivot_rate(simple_df, index="cat", columns="label", target="x", aggfunc="invalid")

    def test_sum_aggfunc(self, simple_df):
        result = pivot_rate(simple_df, index="cat", columns="label", target="x", aggfunc="sum")
        assert isinstance(result, pd.DataFrame)


class TestTopExtremeValues:
    def test_returns_top_k_highest(self, simple_df):
        result = top_extreme_values(simple_df, col="x", k=5)
        assert len(result) == 5
        assert result["x"].is_monotonic_decreasing

    def test_returns_top_k_lowest(self, simple_df):
        result = top_extreme_values(simple_df, col="x", k=5, highest=False)
        assert len(result) == 5
        assert result["x"].is_monotonic_increasing

    def test_missing_column_raises(self, simple_df):
        with pytest.raises(ValueError):
            top_extreme_values(simple_df, col="nonexistent")


# ===========================================================================
# Quality
# ===========================================================================


class TestMissingSummary:
    def test_returns_dataframe(self, simple_df):
        result = missing_summary(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self, simple_df):
        result = missing_summary(simple_df)
        for col in ["column", "dtype", "total", "missing", "missing_pct"]:
            assert col in result.columns

    def test_numeric_columns_have_zero_count(self, simple_df):
        result = missing_summary(simple_df)
        row = result[result["column"] == "x"].iloc[0]
        assert row["zero"] is not None

    def test_categorical_columns_have_null_zero_count(self, simple_df):
        result = missing_summary(simple_df)
        row = result[result["column"] == "cat"].iloc[0]
        assert pd.isna(row["zero"])

    def test_missing_pct_computed_correctly(self):
        df = pd.DataFrame({"a": [1, np.nan, 3, np.nan], "b": ["x", "y", "z", "w"]})
        result = missing_summary(df)
        row = result[result["column"] == "a"].iloc[0]
        assert row["missing_pct"] == pytest.approx(50.0)


class TestMissingHeatmap:
    def test_returns_figure(self):
        df = pd.DataFrame(
            {
                "a": [1, np.nan, 3, np.nan, 5],
                "b": [np.nan, 2, np.nan, 4, 5],
                "c": [1, 2, 3, 4, 5],
            }
        )
        fig = missing_heatmap(df)
        assert isinstance(fig, go.Figure)

    def test_fewer_than_two_missing_columns_raises(self, simple_df):
        with pytest.raises(ValueError, match="2 columns"):
            missing_heatmap(simple_df)  # simple_df has no missing values


class TestDuplicateRows:
    def test_no_duplicates_returns_empty(self, simple_df):
        result = duplicate_rows(simple_df)
        assert result.empty

    def test_detects_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        result = duplicate_rows(df)
        assert len(result) == 2
        assert "_dup_count" in result.columns

    def test_id_cols_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "z"]})
        result = duplicate_rows(df, id_cols=["a"])
        assert len(result) == 2


class TestDuplicateColumns:
    def test_detects_identical_columns(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [4, 5, 6]})
        result = duplicate_columns(df)
        assert len(result) == 1
        assert set(result.iloc[0][["col1", "col2"]]) == {"a", "b"}

    def test_no_duplicates_returns_empty(self, simple_df):
        result = duplicate_columns(simple_df)
        assert result.empty

    def test_partial_match_threshold(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 9]})
        # 3/4 = 75% match
        result_75 = duplicate_columns(df, threshold=75.0)
        result_100 = duplicate_columns(df, threshold=100.0)
        assert len(result_75) == 1
        assert result_100.empty


class TestPsi:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 1000))
        result = psi(s, s.copy())
        assert result < 0.05

    def test_different_distributions_high_psi(self):
        rng = np.random.default_rng(0)
        expected = pd.Series(rng.normal(0, 1, 1000))
        actual = pd.Series(rng.normal(5, 1, 1000))
        result = psi(expected, actual)
        assert result > 0.2

    def test_empty_series_returns_nan(self):
        s = pd.Series([], dtype=float)
        result = psi(s, pd.Series([1.0, 2.0]))
        assert np.isnan(result)

    def test_uniform_method(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 500))
        result = psi(s, s.copy(), method="uniform")
        assert result < 0.1


class TestPsiCategory:
    def test_identical_distributions_near_zero(self):
        s = pd.Series(["A", "B", "C"] * 100)
        result = psi_category(s, s.copy())
        assert result < 0.01

    def test_different_distributions_high_psi(self):
        expected = pd.Series(["A"] * 90 + ["B"] * 10)
        actual = pd.Series(["A"] * 10 + ["B"] * 90)
        result = psi_category(expected, actual)
        assert result > 0.2

    def test_returns_float(self):
        s = pd.Series(["X", "Y"] * 50)
        result = psi_category(s, s.copy())
        assert isinstance(result, float)


class TestPsiReport:
    def test_returns_dataframe(self, simple_df):
        result = psi_report(simple_df, simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_contains_expected_columns(self, simple_df):
        result = psi_report(simple_df, simple_df)
        for col in ["feature", "type", "psi", "stability"]:
            assert col in result.columns

    def test_stable_when_same_data(self, simple_df):
        result = psi_report(simple_df, simple_df)
        assert (result["stability"] == "Stable").all()

    def test_sorted_by_psi_descending(self, simple_df):
        result = psi_report(simple_df, simple_df)
        assert result["psi"].is_monotonic_decreasing

    def test_subset_cols(self, simple_df):
        result = psi_report(simple_df, simple_df, cols=["x", "y"])
        assert set(result["feature"]) == {"x", "y"}


# ---------------------------------------------------------------------------
# Import test – verify public API is accessible
# ---------------------------------------------------------------------------

from src.caketool.eda import (  # noqa: E402, F811 (after all other imports to keep organisation clear)
    EDAConfig,
    bar_category_vs_category,
    box_by_category,
    correlation_heatmap,
    correlation_table,
    cramers_v,
    cramers_v_heatmap,
    cramers_v_target,
    duplicate_columns,
    duplicate_rows,
    histogram_by_label,
    line_with_ma,
    missing_heatmap,
    missing_summary,
    parallel_coordinates,
    pivot_count,
    pivot_rate,
    plot_categorical_frequency,
    plot_numeric_distribution,
    profile,
    psi,
    psi_category,
    psi_report,
    roc_curve_plot,
    scatter,
    scatter_3d,
    scatter_matrix,
    stacked_bar,
    summarize_categorical_series,
    summarize_numeric_series,
    top_extreme_values,
    violin_by_category,
)


def test_public_api_importable():
    assert callable(plot_numeric_distribution)
    assert callable(plot_categorical_frequency)
    assert callable(scatter)
    assert callable(profile)
    assert callable(psi_report)
    assert EDAConfig is not None
