"""EDA (Exploratory Data Analysis) module for caketool.

Supports pandas DataFrames with Plotly as the sole visualization backend.

Quick start
-----------
>>> import pandas as pd
>>> from caketool import eda
>>>
>>> df = pd.read_csv("data.csv")
>>>
>>> # Dataset overview
>>> eda.profile(df)
>>> eda.correlation_heatmap(df)
>>>
>>> # Univariate
>>> eda.histogram(df["age"])
>>> eda.value_counts(df["category"])
>>>
>>> # Bivariate
>>> eda.scatter(df, x="income", y="spend", color_by="segment")
>>> eda.box_by_category(df, cat_col="segment", num_col="income")
>>>
>>> # Data quality
>>> eda.missing_summary(df)
>>> eda.psi_report(df_train, df_test)
"""

# Bivariate
from .bivariate import (
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
from .config import EDAConfig

# Multivariate
from .multivariate import (
    parallel_coordinates,
    scatter_3d,
    scatter_matrix,
    stacked_bar,
)

# Overview
from .overview import (
    correlation_heatmap,
    cramers_v_heatmap,
    pivot_count,
    pivot_rate,
    profile,
    top_extreme_values,
)

# Data quality
from .quality import (
    duplicate_columns,
    duplicate_rows,
    missing_heatmap,
    missing_summary,
    psi,
    psi_category,
    psi_report,
)

# Univariate
from .univariate import (
    bar_count,
    distribution,
    histogram,
    overlay_histogram,
    percentile_table,
    pie_chart,
    value_counts,
)

__all__ = [
    # config
    "EDAConfig",
    # univariate
    "histogram",
    "overlay_histogram",
    "distribution",
    "pie_chart",
    "bar_count",
    "percentile_table",
    "value_counts",
    # bivariate
    "scatter",
    "line_with_ma",
    "box_by_category",
    "violin_by_category",
    "histogram_by_label",
    "bar_category_vs_category",
    "cramers_v",
    "cramers_v_target",
    "correlation_table",
    "roc_curve_plot",
    # multivariate
    "parallel_coordinates",
    "scatter_3d",
    "scatter_matrix",
    "stacked_bar",
    # quality
    "missing_summary",
    "missing_heatmap",
    "duplicate_rows",
    "duplicate_columns",
    "psi",
    "psi_category",
    "psi_report",
    # overview
    "profile",
    "correlation_heatmap",
    "cramers_v_heatmap",
    "pivot_count",
    "pivot_rate",
    "top_extreme_values",
]
