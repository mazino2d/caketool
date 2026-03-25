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
>>> eda.plot_numeric_distribution(df["age"])
>>> eda.plot_categorical_frequency(df["category"])
>>>
>>> # Bivariate
>>> eda.plot_scatter(df, x="income", y="spend", color_by="segment")
>>> eda.plot_distribution_by_group(df, cat_col="segment", num_col="income", mode="box")
>>>
>>> # Data quality
>>> eda.missing_summary(df)
>>> eda.psi_report(df_train, df_test)
"""

# Bivariate
from .bivariate import (
    plot_category_heatmap,
    plot_distribution_by_group,
    plot_roc_curve,
    plot_scatter,
    plot_time_series,
    rank_associations,
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
    plot_categorical_frequency,
    plot_numeric_distribution,
    summarize_categorical_series,
    summarize_numeric_series,
)

__all__ = [
    # config
    "EDAConfig",
    # univariate
    "plot_numeric_distribution",
    "plot_categorical_frequency",
    "summarize_numeric_series",
    "summarize_categorical_series",
    # bivariate
    "plot_scatter",
    "plot_time_series",
    "plot_distribution_by_group",
    "plot_category_heatmap",
    "rank_associations",
    "plot_roc_curve",
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
