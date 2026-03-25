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

# Overview
from .overview import (
    calculate_correlations,
    plot_correlations,
    profile,
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
    # overview
    "profile",
    "calculate_correlations",
    "plot_correlations",
]
