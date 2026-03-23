"""Data quality assessment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._validators import require_columns
from .config import EDAConfig


def _cfg(cfg: EDAConfig | None) -> EDAConfig:
    return cfg if cfg is not None else EDAConfig()


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of missing, zero, and negative values per column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: column, dtype, total, missing, missing_pct,
                 zero (numeric only), zero_pct, negative (numeric only), negative_pct.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = s.isna().sum()
        row: dict = {
            "column": col,
            "dtype": str(s.dtype),
            "total": n,
            "missing": missing,
            "missing_pct": round(missing / n * 100, 2) if n else 0,
        }
        if pd.api.types.is_numeric_dtype(s):
            row["zero"] = (s == 0).sum()
            row["zero_pct"] = round(row["zero"] / n * 100, 2) if n else 0
            row["negative"] = (s < 0).sum()
            row["negative_pct"] = round(row["negative"] / n * 100, 2) if n else 0
        else:
            row["zero"] = None
            row["zero_pct"] = None
            row["negative"] = None
            row["negative_pct"] = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)


def missing_heatmap(
    df: pd.DataFrame,
    mask_threshold: float = 0.0,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Hierarchically clustered heatmap of column missingness correlation.

    Computes pairwise correlation of missing-value indicators and clusters
    columns so that columns that tend to be missing together are grouped.

    Parameters
    ----------
    df : pd.DataFrame
    mask_threshold : float
        Hide cells with |correlation| below this value.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    c = _cfg(cfg)
    miss = df.isnull().astype(int)
    # Keep only columns that have some missing values
    miss = miss.loc[:, miss.sum() > 0]
    if miss.shape[1] < 2:
        raise ValueError("Need at least 2 columns with missing values to plot heatmap.")
    corr = miss.corr()
    order = leaves_list(linkage(corr.values, method="complete"))
    corr_reordered = corr.iloc[order, order]
    z = corr_reordered.values.copy()
    if mask_threshold > 0:
        z[np.abs(z) < mask_threshold] = np.nan
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=corr_reordered.columns.tolist(),
            y=corr_reordered.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar={"title": "Corr"},
        )
    )
    fig.update_layout(
        title="Missing Value Correlation Heatmap (Clustered)",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


def duplicate_rows(
    df: pd.DataFrame,
    id_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summary of duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
    id_cols : list[str], optional
        Columns to use as identity key. If None, uses all columns.

    Returns
    -------
    pd.DataFrame
        Duplicated rows with an added ``_dup_count`` column.
        Returns an empty DataFrame if no duplicates found.
    """
    subset = id_cols or df.columns.tolist()
    if id_cols:
        require_columns(df, id_cols)
    dup_mask = df.duplicated(subset=subset, keep=False)
    result = df[dup_mask].copy()
    if not result.empty:
        row_hashes = pd.util.hash_pandas_object(result[subset], index=False)
        result["_dup_count"] = row_hashes.map(row_hashes.value_counts())
        result = result.sort_values("_dup_count", ascending=False)
    return result.reset_index(drop=True)


def duplicate_columns(
    df: pd.DataFrame,
    threshold: float = 100.0,
) -> pd.DataFrame:
    """Detect columns that are identical or nearly identical.

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Minimum percentage match (0–100) to flag as duplicate. Default 100.

    Returns
    -------
    pd.DataFrame
        Columns: col1, col2, match_pct — sorted by match_pct desc.
    """
    n = len(df)
    cols = df.columns.tolist()
    rows = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            match = (df[c1].astype(object) == df[c2].astype(object)).sum()
            pct = round(match / n * 100, 2) if n else 0
            if pct >= threshold:
                rows.append({"col1": c1, "col2": c2, "match_pct": pct})
    result = pd.DataFrame(rows, columns=["col1", "col2", "match_pct"] if not rows else None)
    if result.empty:
        return result
    return result.sort_values("match_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------------


def _sub_psi(e: float, a: float) -> float:
    """PSI contribution for a single bucket."""
    eps = 1e-6
    e = max(e, eps)
    a = max(a, eps)
    return (a - e) * np.log(a / e)


def psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10,
    method: str = "quantile",
) -> float:
    """Compute Population Stability Index (PSI) for a numeric series.

    PSI < 0.1  → no significant change
    PSI < 0.2  → moderate change
    PSI ≥ 0.2  → significant change

    Parameters
    ----------
    expected : pd.Series
        Reference distribution (e.g. training set).
    actual : pd.Series
        New distribution (e.g. scoring set).
    buckets : int
        Number of bins.
    method : str
        "quantile" bins by quantiles of *expected*; "uniform" uses equal-width bins.

    Returns
    -------
    float
        PSI value.
    """
    e = expected.dropna().values.astype(float)
    a = actual.dropna().values.astype(float)
    if len(e) == 0 or len(a) == 0:
        return np.nan
    if method == "quantile":
        breakpoints = np.quantile(e, np.linspace(0, 1, buckets + 1))
    else:
        breakpoints = np.linspace(e.min(), e.max(), buckets + 1)
    breakpoints = np.unique(breakpoints)
    e_pcts = np.histogram(e, bins=breakpoints)[0] / len(e)
    a_pcts = np.histogram(a, bins=breakpoints)[0] / len(a)
    return float(sum(_sub_psi(ep, ap) for ep, ap in zip(e_pcts, a_pcts, strict=True)))


def psi_category(
    expected: pd.Series,
    actual: pd.Series,
) -> float:
    """Compute PSI for a categorical series.

    Parameters
    ----------
    expected : pd.Series
        Reference distribution.
    actual : pd.Series
        New distribution.

    Returns
    -------
    float
        PSI value.
    """
    e_counts = expected.value_counts(normalize=True)
    a_counts = actual.value_counts(normalize=True)
    all_cats = e_counts.index.union(a_counts.index)
    total = 0.0
    for cat in all_cats:
        e_p = e_counts.get(cat, 1e-6)
        a_p = a_counts.get(cat, 1e-6)
        total += _sub_psi(e_p, a_p)
    return float(total)


def psi_report(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols: list[str] | None = None,
    buckets: int = 10,
    cat_threshold: int = 20,
) -> pd.DataFrame:
    """Compute PSI for all specified columns between train and test sets.

    Parameters
    ----------
    df_train : pd.DataFrame
        Reference (training) dataset.
    df_test : pd.DataFrame
        New (scoring/test) dataset.
    cols : list[str], optional
        Columns to check. Defaults to all shared columns.
    buckets : int
        Buckets for numeric PSI.
    cat_threshold : int
        Columns with ≤ this many unique values are treated as categorical.

    Returns
    -------
    pd.DataFrame
        Columns: feature, type, psi, stability — sorted by psi descending.
    """
    shared = list(set(df_train.columns) & set(df_test.columns))
    cols = cols or shared
    rows = []
    for col in cols:
        e = df_train[col].dropna()
        a = df_test[col].dropna()
        if pd.api.types.is_numeric_dtype(e) and e.nunique() > cat_threshold:
            val = psi(e, a, buckets=buckets)
            kind = "numeric"
        else:
            val = psi_category(e.astype(str), a.astype(str))
            kind = "categorical"
        if np.isnan(val):
            stability = "N/A"
        elif val < 0.1:
            stability = "Stable"
        elif val < 0.2:
            stability = "Moderate"
        else:
            stability = "Unstable"
        rows.append({"feature": col, "type": kind, "psi": round(val, 4), "stability": stability})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
