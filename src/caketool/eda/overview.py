"""Dataset-level overview and summary statistics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..metric.association_metric import association
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
                    "zero": int((s == 0).sum()),
                    "zero_pct": round((s == 0).sum() / n * 100, 2) if n else 0,
                    "negative": int((s < 0).sum()),
                    "negative_pct": round((s < 0).sum() / n * 100, 2) if n else 0,
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
                    "zero": None,
                    "zero_pct": None,
                    "negative": None,
                    "negative_pct": None,
                    "top_value": vc.index[0] if len(vc) else None,
                    "top_freq": vc.iloc[0] if len(vc) else None,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation / association matrix
# ---------------------------------------------------------------------------


def calculate_correlations(
    df: pd.DataFrame,
    num_method: Literal["pearson", "spearman"] = "pearson",
    unique_threshold: int = 50,
) -> pd.DataFrame:
    """Compute pairwise association between all column pairs in a DataFrame.

    Automatically selects the metric based on column types:

    - numeric × numeric  : Pearson or Spearman (``num_method``), absolute value.
    - numeric × categorical : eta coefficient.
    - categorical × categorical : Cramér\'s V.

    All values are in [0, 1]. Diagonal = 1.0.

    Parameters
    ----------
    df : pd.DataFrame
    num_method : {"pearson", "spearman"}
        Method for numeric-numeric pairs. Default: ``"pearson"``.
    unique_threshold : int
        Columns with more unique values than this threshold are treated as
        numeric. Default: 50.

    Returns
    -------
    pd.DataFrame
        Symmetric N×N association matrix with column names as index and columns.
    """
    cols = df.columns.tolist()
    n = len(cols)
    is_num = {
        col: pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > unique_threshold for col in cols
    }

    mat = np.eye(n)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            idx = df[c1].notna() & df[c2].notna()
            if idx.sum() < 2:
                mat[i, j] = mat[j, i] = np.nan
                continue
            a, b = df.loc[idx, c1], df.loc[idx, c2]
            if is_num[c1] and is_num[c2]:
                val, _ = association(a, b, method=num_method)
                val = abs(val)
            elif not is_num[c1] and is_num[c2]:
                val, _ = association(a, b, method="eta")
            elif is_num[c1] and not is_num[c2]:
                val, _ = association(b, a, method="eta")
            else:
                val, _ = association(a, b, method="cramers_v")
            mat[i, j] = mat[j, i] = round(val, 4)

    return pd.DataFrame(mat, index=cols, columns=cols)


def plot_correlations(
    df: pd.DataFrame,
    num_method: Literal["pearson", "spearman"] = "pearson",
    unique_threshold: int = 50,
    mask_threshold: float = 0.0,
    cluster: bool = True,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Heatmap of pairwise associations across all columns.

    Automatically selects association metric per column-pair type (see
    :func:`calculate_all_correlations`). All values normalised to [0, 1].

    Parameters
    ----------
    df : pd.DataFrame
    num_method : {"pearson", "spearman"}
        Method for numeric-numeric pairs. Default: ``"pearson"``.
    unique_threshold : int
        Columns with more unique values than this threshold are treated as
        numeric. Default: 50.
    mask_threshold : float
        Hide cells with value below this threshold. Default: 0.0.
    cluster : bool
        Reorder columns by hierarchical clustering. Default: True.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    from scipy.cluster.hierarchy import leaves_list, linkage

    c = _cfg(cfg)
    corr = calculate_correlations(df, num_method, unique_threshold=unique_threshold)
    if corr.shape[1] < 2:
        raise ValueError("Need at least 2 columns to compute associations.")
    if cluster:
        order = leaves_list(linkage(corr.fillna(0).values, method="complete"))
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
            colorbar={"title": "Association"},
        )
    )
    fig.update_layout(
        title=f"Association Heatmap ({num_method})" + (" — Clustered" if cluster else ""),
        template=c.template,
        width=max(c.width, corr.shape[1] * 50),
        height=max(c.height, corr.shape[1] * 50),
    )
    return fig
