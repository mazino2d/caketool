"""Two-variable (bivariate) analysis and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

from ._validators import (
    clip_quantiles,
    require_column,
    require_columns,
    require_numeric,
    top_k_series,
)
from .config import EDAConfig


def _cfg(cfg: EDAConfig | None) -> EDAConfig:
    return cfg if cfg is not None else EDAConfig()


# ---------------------------------------------------------------------------
# Numeric vs Numeric
# ---------------------------------------------------------------------------


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str | None = None,
    sample_n: int | None = 5000,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Scatter plot of two numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        Column name for x-axis.
    y : str
        Column name for y-axis.
    color_by : str, optional
        Column name for color grouping.
    sample_n : int, optional
        Randomly sample this many rows to avoid overplotting. None = all rows.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [x, y] + ([color_by] if color_by else []))
    c = _cfg(cfg)
    data = df[[x, y] + ([color_by] if color_by else [])].dropna()
    if sample_n and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42)

    fig = go.Figure()
    if color_by:
        for i, grp in enumerate(sorted(data[color_by].unique())):
            sub = data[data[color_by] == grp]
            fig.add_trace(
                go.Scatter(
                    x=sub[x],
                    y=sub[y],
                    mode="markers",
                    name=str(grp),
                    marker={"color": c.color_palette[i % len(c.color_palette)], "opacity": 0.6, "size": 5},
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=data[x], y=data[y], mode="markers", marker={"color": c.color_palette[0], "opacity": 0.6, "size": 5}
            )
        )

    fig.update_layout(
        title=f"{x} vs {y}",
        xaxis_title=x,
        yaxis_title=y,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def line_with_ma(
    df: pd.DataFrame,
    x: str,
    y: str,
    ma: int = 7,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Line chart with optional moving average overlay.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        x-axis column (typically time).
    y : str
        Numeric column.
    ma : int
        Moving average window. Set to 0 to disable.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [x, y])
    c = _cfg(cfg)
    data = df[[x, y]].dropna().sort_values(x)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data[x], y=data[y], mode="lines", name=y, line={"color": c.color_palette[0], "width": 1.5})
    )
    if ma > 0:
        ma_vals = data[y].rolling(ma, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=ma_vals,
                mode="lines",
                name=f"MA({ma})",
                line={"color": c.color_palette[1], "width": 2, "dash": "dash"},
            )
        )
    fig.update_layout(
        title=f"{y} over {x}",
        xaxis_title=x,
        yaxis_title=y,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Categorical vs Numeric
# ---------------------------------------------------------------------------


def box_by_category(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    top_k: int | None = None,
    orientation: str = "v",
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Box plots of a numeric column grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    cat_col : str
        Categorical column name.
    num_col : str
        Numeric column name.
    top_k : int, optional
        Limit to top K categories by count.
    orientation : str
        "v" (vertical) or "h" (horizontal).
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [cat_col, num_col])
    require_numeric(df[num_col])
    c = _cfg(cfg)
    data = df[[cat_col, num_col]].dropna()
    cats = data[cat_col].value_counts().head(top_k or c.top_k_categories).index
    data = data[data[cat_col].isin(cats)]
    fig = go.Figure()
    for i, cat in enumerate(cats):
        sub = data[data[cat_col] == cat][num_col]
        kwargs = {"y": sub, "name": str(cat)} if orientation == "v" else {"x": sub, "name": str(cat)}
        fig.add_trace(go.Box(**kwargs, marker_color=c.color_palette[i % len(c.color_palette)]))
    fig.update_layout(
        title=f"{num_col} by {cat_col}",
        xaxis_title=cat_col if orientation == "v" else num_col,
        yaxis_title=num_col if orientation == "v" else cat_col,
        showlegend=False,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def violin_by_category(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    top_k: int | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Violin plots of a numeric column grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    cat_col : str
        Categorical column.
    num_col : str
        Numeric column.
    top_k : int, optional
        Limit to top K categories.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [cat_col, num_col])
    require_numeric(df[num_col])
    c = _cfg(cfg)
    data = df[[cat_col, num_col]].dropna()
    cats = data[cat_col].value_counts().head(top_k or c.top_k_categories).index
    data = data[data[cat_col].isin(cats)]
    fig = go.Figure()
    for i, cat in enumerate(cats):
        sub = data[data[cat_col] == cat][num_col]
        fig.add_trace(
            go.Violin(
                y=sub,
                name=str(cat),
                box_visible=True,
                meanline_visible=True,
                fillcolor=c.color_palette[i % len(c.color_palette)],
                opacity=0.7,
                line_color="black",
            )
        )
    fig.update_layout(
        title=f"{num_col} by {cat_col} (violin)",
        xaxis_title=cat_col,
        yaxis_title=num_col,
        showlegend=False,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def histogram_by_label(
    df: pd.DataFrame,
    feature: str,
    label: str,
    nbins: int = 30,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
    normalize: bool = True,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Overlaid histograms of a feature split by a binary/categorical label.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Numeric feature column.
    label : str
        Label column (binary or categorical).
    nbins : int
        Number of bins.
    low_trim : float
        Lower quantile clip.
    high_trim : float
        Upper quantile clip.
    normalize : bool
        If True, use probability density (so classes of different sizes are comparable).
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [feature, label])
    require_numeric(df[feature])
    c = _cfg(cfg)
    data = df[[feature, label]].dropna()
    fig = go.Figure()
    for i, grp in enumerate(sorted(data[label].unique())):
        s = clip_quantiles(data[data[label] == grp][feature], low_trim, high_trim)
        fig.add_trace(
            go.Histogram(
                x=s,
                nbinsx=nbins,
                name=str(grp),
                histnorm="probability density" if normalize else "",
                marker_color=c.color_palette[i % len(c.color_palette)],
                opacity=0.65,
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=f"{feature} by {label}",
        xaxis_title=feature,
        yaxis_title="Density" if normalize else "Count",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Categorical vs Categorical
# ---------------------------------------------------------------------------


def bar_category_vs_category(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    normalize: bool = False,
    top_k1: int | None = None,
    top_k2: int | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Grouped bar chart comparing two categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    col1 : str
        x-axis categories.
    col2 : str
        Color-grouped categories.
    normalize : bool
        If True, show row-percentage instead of counts.
    top_k1 : int, optional
        Limit col1 to top K categories.
    top_k2 : int, optional
        Limit col2 to top K categories.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [col1, col2])
    c = _cfg(cfg)
    data = df[[col1, col2]].dropna()
    if top_k1:
        data[col1] = top_k_series(data[col1], top_k1)
    if top_k2:
        data[col2] = top_k_series(data[col2], top_k2)
    ct = pd.crosstab(data[col1], data[col2], normalize="index" if normalize else False)
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_trace(
            go.Bar(
                x=ct.index.astype(str),
                y=ct[col] * (100 if normalize else 1),
                name=str(col),
                marker_color=c.color_palette[i % len(c.color_palette)],
            )
        )
    fig.update_layout(
        barmode="group",
        title=f"{col1} vs {col2}",
        xaxis_title=col1,
        yaxis_title="Percentage (%)" if normalize else "Count",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Statistical associations
# ---------------------------------------------------------------------------


def cramers_v(series1: pd.Series, series2: pd.Series) -> float:
    """Compute Cramér's V association between two categorical series.

    Parameters
    ----------
    series1 : pd.Series
    series2 : pd.Series

    Returns
    -------
    float
        Cramér's V in [0, 1].
    """
    ct = pd.crosstab(series1, series2)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    k = min(ct.shape) - 1
    if n == 0 or k == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def cramers_v_target(
    df: pd.DataFrame,
    target: str,
    cat_cols: list[str] | None = None,
    unique_threshold: int = 50,
) -> pd.DataFrame:
    """Compute Cramér's V between each categorical column and a target.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Target column.
    cat_cols : list[str], optional
        Columns to test. If None, auto-detects columns with ≤ *unique_threshold* unique values.
    unique_threshold : int
        Max unique values to consider a column categorical when *cat_cols* is None.

    Returns
    -------
    pd.DataFrame
        Columns: feature, cramers_v, sorted descending.
    """
    require_column(df, target)
    if cat_cols is None:
        cat_cols = [c for c in df.columns if c != target and df[c].nunique() <= unique_threshold]
    results = []
    for col in cat_cols:
        s1 = df[col].astype(str)
        s2 = df[target].astype(str)
        mask = s1.notna() & s2.notna()
        if mask.sum() < 2:
            continue
        v = cramers_v(s1[mask], s2[mask])
        results.append({"feature": col, "cramers_v": round(v, 4)})
    return pd.DataFrame(results).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def correlation_table(
    df: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Pairwise correlation table filtered by absolute threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of numeric columns.
    method : str
        "pearson", "spearman", or "kendall".
    threshold : float
        Only include pairs with |corr| >= threshold.

    Returns
    -------
    pd.DataFrame
        Columns: col1, col2, correlation, sorted by absolute value.
    """
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr(method=method)
    rows = []
    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            val = corr.loc[c1, c2]
            if abs(val) >= threshold:
                rows.append({"col1": c1, "col2": c2, "correlation": round(val, 4)})
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def roc_curve_plot(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """ROC curve with AUC annotation.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Binary (0/1) label column.
    pred_col : str
        Predicted probability column.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    from sklearn.metrics import auc, roc_curve

    require_columns(df, [label_col, pred_col])
    c = _cfg(cfg)
    data = df[[label_col, pred_col]].dropna()
    fpr, tpr, _ = roc_curve(data[label_col], data[pred_col])
    auc_val = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_val:.4f})", line={"color": c.color_palette[0], "width": 2}
        )
    )
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line={"color": "gray", "dash": "dash"}))
    fig.update_layout(
        title=f"ROC Curve — AUC = {auc_val:.4f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig
