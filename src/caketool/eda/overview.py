"""Dataset-level overview and summary statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._validators import require_column, require_columns
from .bivariate import cramers_v
from .config import EDAConfig


def _cfg(cfg: EDAConfig | None) -> EDAConfig:
    return cfg if cfg is not None else EDAConfig()


# ---------------------------------------------------------------------------
# Dataset profile
# ---------------------------------------------------------------------------


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive per-column summary of a DataFrame.

    Returns dtype, cardinality, missing %, and basic stats (numeric cols)
    or top value / frequency (categorical cols).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per column with descriptive statistics.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = s.isna().sum()
        row: dict = {
            "column": col,
            "dtype": str(s.dtype),
            "n_total": n,
            "n_missing": missing,
            "missing_pct": round(missing / n * 100, 2) if n else 0,
            "n_unique": s.nunique(dropna=True),
            "unique_pct": round(s.nunique(dropna=True) / n * 100, 2) if n else 0,
        }
        if pd.api.types.is_numeric_dtype(s):
            valid = s.dropna()
            row.update(
                {
                    "mean": round(valid.mean(), 4) if len(valid) else None,
                    "std": round(valid.std(), 4) if len(valid) else None,
                    "min": valid.min() if len(valid) else None,
                    "p25": valid.quantile(0.25) if len(valid) else None,
                    "p50": valid.quantile(0.50) if len(valid) else None,
                    "p75": valid.quantile(0.75) if len(valid) else None,
                    "max": valid.max() if len(valid) else None,
                    "skewness": round(valid.skew(), 4) if len(valid) else None,
                    "kurtosis": round(valid.kurtosis(), 4) if len(valid) else None,
                    "top_value": None,
                    "top_freq": None,
                }
            )
        else:
            vc = s.value_counts(dropna=True)
            row.update(
                {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "max": None,
                    "skewness": None,
                    "kurtosis": None,
                    "top_value": vc.index[0] if len(vc) else None,
                    "top_freq": vc.iloc[0] if len(vc) else None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation heatmaps
# ---------------------------------------------------------------------------


def correlation_heatmap(
    df: pd.DataFrame,
    method: str = "pearson",
    mask_threshold: float = 0.0,
    cluster: bool = True,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Hierarchically clustered correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    method : str
        "pearson", "spearman", or "kendall".
    mask_threshold : float
        Hide cells with |correlation| below this value.
    cluster : bool
        If True, reorder columns by hierarchical clustering.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    c = _cfg(cfg)
    num_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
    if num_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns to compute correlation.")
    corr = num_df.corr(method=method)
    if cluster:
        order = leaves_list(linkage(corr.fillna(0).values, method="complete"))
        corr = corr.iloc[order, order]
    z = corr.values.copy()
    if mask_threshold > 0:
        z[np.abs(z) < mask_threshold] = np.nan
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(z, 2),
            texttemplate="%{text}",
            colorbar={"title": method.capitalize()},
        )
    )
    fig.update_layout(
        title=f"Correlation Heatmap ({method})" + (" — Clustered" if cluster else ""),
        template=c.template,
        width=max(c.width, num_df.shape[1] * 50),
        height=max(c.height, num_df.shape[1] * 50),
    )
    return fig


def cramers_v_heatmap(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    unique_threshold: int = 50,
    mask_threshold: float = 0.0,
    cluster: bool = True,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Cramér's V pairwise association heatmap for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    cat_cols : list[str], optional
        Columns to include. Auto-detects if None.
    unique_threshold : int
        Max unique values to consider a column categorical.
    mask_threshold : float
        Hide cells below this threshold.
    cluster : bool
        Reorder by hierarchical clustering.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    c = _cfg(cfg)
    if cat_cols is None:
        cat_cols = [col for col in df.columns if df[col].nunique() <= unique_threshold]
    if len(cat_cols) < 2:
        raise ValueError("Need at least 2 categorical columns to compute Cramér's V.")
    # Build matrix
    mat = np.zeros((len(cat_cols), len(cat_cols)))
    for i, c1 in enumerate(cat_cols):
        for j, c2 in enumerate(cat_cols):
            if i == j:
                mat[i, j] = 1.0
            elif i < j:
                mask = df[c1].notna() & df[c2].notna()
                val = cramers_v(df.loc[mask, c1].astype(str), df.loc[mask, c2].astype(str))
                mat[i, j] = val
                mat[j, i] = val
    corr = pd.DataFrame(mat, index=cat_cols, columns=cat_cols)
    if cluster and len(cat_cols) > 2:
        order = leaves_list(linkage(corr.values, method="complete"))
        corr = corr.iloc[order, order]
    z = corr.values.copy()
    if mask_threshold > 0:
        z[z < mask_threshold] = np.nan
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=np.round(z, 2),
            texttemplate="%{text}",
            colorbar={"title": "Cramér's V"},
        )
    )
    fig.update_layout(
        title="Cramér's V Association Heatmap" + (" — Clustered" if cluster else ""),
        template=c.template,
        width=max(c.width, len(cat_cols) * 50),
        height=max(c.height, len(cat_cols) * 50),
    )
    return fig


# ---------------------------------------------------------------------------
# Pivot tables
# ---------------------------------------------------------------------------


def pivot_count(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str | None = None,
    margins: bool = True,
) -> pd.DataFrame:
    """Pivot table with count aggregation.

    Parameters
    ----------
    df : pd.DataFrame
    index : str
        Row grouping column.
    columns : str
        Column grouping column.
    values : str, optional
        Column to count. Defaults to *index*.
    margins : bool
        Add row/column totals.

    Returns
    -------
    pd.DataFrame
    """
    require_columns(df, [index, columns])
    return pd.crosstab(df[index], df[columns], margins=margins, margins_name="Total")


def pivot_rate(
    df: pd.DataFrame,
    index: str,
    columns: str,
    target: str,
    aggfunc: str = "mean",
    margins: bool = True,
) -> pd.DataFrame:
    """Pivot table with rate aggregation (e.g. bad rate, conversion rate).

    Parameters
    ----------
    df : pd.DataFrame
    index : str
        Row grouping column.
    columns : str
        Column grouping column.
    target : str
        Numeric target column to aggregate.
    aggfunc : str
        Aggregation function: "mean", "sum", "median".
    margins : bool
        Add totals.

    Returns
    -------
    pd.DataFrame
        Values rounded to 4 decimal places.
    """
    require_columns(df, [index, columns, target])
    funcs = {"mean": "mean", "sum": "sum", "median": "median"}
    if aggfunc not in funcs:
        raise ValueError(f"aggfunc must be one of {list(funcs)}.")
    result = df.pivot_table(
        index=index,
        columns=columns,
        values=target,
        aggfunc=funcs[aggfunc],
        observed=False,
        margins=margins,
        margins_name="Total",
    )
    return result.round(4)


# ---------------------------------------------------------------------------
# Extreme values
# ---------------------------------------------------------------------------


def top_extreme_values(
    df: pd.DataFrame,
    col: str,
    k: int = 10,
    highest: bool = True,
) -> pd.DataFrame:
    """Return the top K rows by extreme value in a column.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to rank by.
    k : int
        Number of rows to return.
    highest : bool
        If True, return highest values; if False, return lowest.

    Returns
    -------
    pd.DataFrame
    """
    require_column(df, col)
    return df.nlargest(k, col) if highest else df.nsmallest(k, col)
