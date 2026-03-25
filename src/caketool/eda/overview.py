"""Dataset-level overview and summary statistics."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..metric.association_metric import association
from ._validators import require_columns, require_numeric
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


def summarize_missing_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missingness per column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        One row per column with missing and present statistics.
    """
    n_total = len(df)
    rows: list[dict[str, object]] = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        n_present = n_total - n_missing
        rows.append(
            {
                "column": col,
                "n_total": n_total,
                "n_missing": n_missing,
                "missing_pct": round(n_missing / n_total * 100, 2) if n_total else 0.0,
                "n_present": n_present,
                "present_pct": round(n_present / n_total * 100, 2) if n_total else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["column", "n_total", "n_missing", "missing_pct", "n_present", "present_pct"])
    result = pd.DataFrame(rows)
    return result.sort_values(["n_missing", "column"], ascending=[False, True], ignore_index=True)


def summarize_missing_by_row(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize row-level missingness distribution.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Distribution of number of missing columns per row.
    """
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "n_missing_columns",
                "n_rows",
                "row_pct",
                "cumulative_rows",
                "cumulative_pct",
            ]
        )

    row_missing_counts = df.isna().sum(axis=1)
    dist = row_missing_counts.value_counts().sort_index()
    result = pd.DataFrame(
        {
            "n_missing_columns": dist.index.astype(int),
            "n_rows": dist.values.astype(int),
        }
    )
    result["row_pct"] = (result["n_rows"] / len(df) * 100).round(2)
    result["cumulative_rows"] = result["n_rows"].cumsum()
    result["cumulative_pct"] = (result["cumulative_rows"] / len(df) * 100).round(2)
    return result.reset_index(drop=True)


def summarize_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: Literal["tukey", "zscore"] = "tukey",
    tukey_k: float = 1.5,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """Summarize outliers for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str], optional
        Numeric columns to analyze. If omitted, all numeric columns are used.
    method : {"tukey", "zscore"}
        Outlier detection method.
    tukey_k : float
        IQR multiplier when ``method='tukey'``.
    z_threshold : float
        Absolute z-score threshold when ``method='zscore'``.

    Returns
    -------
    pd.DataFrame
        One row per analyzed column with outlier statistics and bounds.
    """
    if method not in {"tukey", "zscore"}:
        raise ValueError("method must be one of {'tukey', 'zscore'}")
    if not isinstance(tukey_k, int | float) or isinstance(tukey_k, bool) or tukey_k <= 0:
        raise ValueError("tukey_k must be a positive number")
    if not isinstance(z_threshold, int | float) or isinstance(z_threshold, bool) or z_threshold <= 0:
        raise ValueError("z_threshold must be a positive number")

    if columns is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        require_columns(df, columns)
        numeric_cols = columns

    rows: list[dict[str, object]] = []
    n_total = len(df)
    for col in numeric_cols:
        series = df[col]
        require_numeric(series)
        valid = series.dropna().astype(float)
        n_valid = int(len(valid))

        if n_valid == 0:
            lower_bound = np.nan
            upper_bound = np.nan
            n_outlier = 0
        elif method == "tukey":
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - tukey_k * iqr
            upper_bound = q3 + tukey_k * iqr
            n_outlier = int(((valid < lower_bound) | (valid > upper_bound)).sum())
        else:
            mean = float(valid.mean())
            std = float(valid.std(ddof=0))
            lower_bound = mean - z_threshold * std
            upper_bound = mean + z_threshold * std
            if std == 0 or np.isnan(std):
                n_outlier = 0
            else:
                n_outlier = int(((valid < lower_bound) | (valid > upper_bound)).sum())

        rows.append(
            {
                "column": col,
                "method": method,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_outlier": n_outlier,
                "outlier_pct": round(n_outlier / n_valid * 100, 2) if n_valid else 0.0,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    return pd.DataFrame(rows)


def summarize_duplicates(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    """Summarize duplicate rows for full-row and subset-key scopes.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list[str], optional
        Columns used as duplicate keys. If omitted, full rows are compared.

    Returns
    -------
    pd.DataFrame
        Single-row summary with duplicate counts and percentages.
    """
    if subset is not None:
        if len(subset) == 0:
            raise ValueError("subset must contain at least one column")
        require_columns(df, subset)

    duplicate_mask = df.duplicated(subset=subset, keep=False)
    n_total = len(df)
    n_duplicate_rows = int(duplicate_mask.sum())

    if n_duplicate_rows == 0:
        n_duplicate_groups = 0
    elif subset is None:
        grouped = df.loc[duplicate_mask].groupby(list(df.columns), dropna=False, observed=False).size()
        n_duplicate_groups = int((grouped > 1).sum())
    else:
        grouped = df.loc[duplicate_mask].groupby(subset, dropna=False, observed=False).size()
        n_duplicate_groups = int((grouped > 1).sum())

    key_columns = subset if subset is not None else list(df.columns)
    scope = "subset" if subset is not None else "full_row"

    return pd.DataFrame(
        [
            {
                "scope": scope,
                "key_columns": ", ".join(key_columns),
                "n_total": n_total,
                "n_duplicate_rows": n_duplicate_rows,
                "duplicate_row_pct": round(n_duplicate_rows / n_total * 100, 2) if n_total else 0.0,
                "n_duplicate_groups": n_duplicate_groups,
            }
        ]
    )


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
