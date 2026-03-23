from __future__ import annotations

import pandas as pd


def require_column(df: pd.DataFrame, col: str) -> None:
    """Raise ValueError if *col* is not in *df*."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available columns: {list(df.columns)}")


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    """Raise ValueError if any column in *cols* is missing from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")


def require_numeric(series: pd.Series) -> None:
    """Raise TypeError if *series* is not numeric."""
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"Column '{series.name}' must be numeric, got dtype '{series.dtype}'.")


def require_nonempty(series: pd.Series) -> None:
    """Raise ValueError if *series* has no non-null values."""
    if series.dropna().empty:
        raise ValueError(f"Column '{series.name}' has no non-null values.")


def clip_quantiles(series: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    """Clip *series* to the [low, high] quantile range.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    low : float
        Lower quantile (0.0–1.0).
    high : float
        Upper quantile (0.0–1.0).

    Returns
    -------
    pd.Series
        Clipped series.
    """
    if low == 0.0 and high == 1.0:
        return series
    lo = series.quantile(low)
    hi = series.quantile(high)
    return series.clip(lo, hi)


def top_k_series(series: pd.Series, k: int, other_label: str = "Others") -> pd.Series:
    """Replace low-frequency categories with *other_label*, keeping top *k*.

    Parameters
    ----------
    series : pd.Series
        Categorical series.
    k : int
        Number of top categories to keep.
    other_label : str
        Label for remaining categories.

    Returns
    -------
    pd.Series
        Series with rare categories replaced.
    """
    top = series.value_counts().nlargest(k).index
    result = series.astype(object).where(series.isin(top), other=other_label)
    if hasattr(series, "cat"):
        return result.astype("category")
    return result
