"""Two-variable (bivariate) analysis and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway, linregress, pearsonr, spearmanr

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


def _interpret_correlation(r: float, p_val: float) -> str:
    """Interpret correlation strength and significance.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient
    p_val : float
        P-value from correlation test

    Returns
    -------
    str
        Interpretation text
    """
    abs_r = abs(r)

    # Strength
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    # Significance
    sig = "significant" if p_val < 0.05 else "NOT significant"

    return f"{strength}, {sig}"


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str | None = None,
    sample_n: int | None = 5000,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Scatter plot of two numeric columns with OLS trend line and correlation annotation.

    Always displays OLS regression trend line and Pearson correlation strength/significance
    in the title (negligible/weak/moderate/strong/very strong + p-value).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    x : str
        Column name for x-axis (numeric).
    y : str
        Column name for y-axis (numeric).
    color_by : str, optional
        Column name for color grouping (creates multiple traces).
    sample_n : int, optional
        Randomly sample this many rows to avoid overplotting. None = all rows. Default: 5000.
    cfg : EDAConfig, optional
        Visualization config.

    Returns
    -------
    go.Figure
        Plotly scatter plot with neon green trend line and correlation interpretation.

    Examples
    --------
    >>> fig = plot_scatter(df, x="age", y="income", color_by="education")
    >>> fig.show()
    """
    require_columns(df, [x, y] + ([color_by] if color_by else []))
    require_numeric(df[x])
    require_numeric(df[y])
    c = _cfg(cfg)

    data = df[[x, y] + ([color_by] if color_by else [])].dropna()
    if sample_n and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42)

    # Always calculate and display correlation
    corr_text = ""
    if len(data) > 2:
        r, p_val = pearsonr(data[x], data[y])
        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        interp_text = _interpret_correlation(r, p_val)
        corr_text = f" | r={r:.3f} ({p_str}) | {interp_text}"

    fig = go.Figure()

    # Plot scatter
    if color_by:
        for i, grp in enumerate(sorted(data[color_by].unique())):
            sub = data[data[color_by] == grp]
            fig.add_trace(
                go.Scatter(
                    x=sub[x],
                    y=sub[y],
                    mode="markers",
                    name=str(grp),
                    marker={
                        "color": c.color_palette[i % len(c.color_palette)],
                        "opacity": 0.6,
                        "size": 5,
                    },
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=data[y],
                mode="markers",
                marker={"color": c.color_palette[0], "opacity": 0.6, "size": 5},
            )
        )

    # Always add OLS trend line with neon green color
    if len(data) > 1:
        slope, intercept, _, _, _ = linregress(data[x], data[y])
        x_range = np.array([data[x].min(), data[x].max()])
        y_trend = slope * x_range + intercept
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_trend,
                mode="lines",
                name="OLS trend",
                line={"color": "#39FF14", "width": 3.5, "dash": "dash"},
                showlegend=True,
            )
        )

    title = f"{x} vs {y}{corr_text}"
    fig.update_layout(
        title=title,
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


def _calculate_eta(cat_series: pd.Series, num_series: pd.Series) -> tuple[float, str, float]:
    """Calculate eta coefficient (correlation ratio) for categorical-numeric association.

    Eta ranges from 0 to 1, measuring how much of the numeric variance
    is explained by the categorical grouping. Also computes ANOVA p-value.

    Parameters
    ----------
    cat_series : pd.Series
        Categorical variable
    num_series : pd.Series
        Numeric variable

    Returns
    -------
    tuple[float, str, float]
        (eta_value, interpretation_text, p_value)
    """
    # Grand mean
    grand_mean = num_series.mean()

    # Between-group sum of squares
    ss_between = 0
    for cat in cat_series.unique():
        group = num_series[cat_series == cat]
        group_mean = group.mean()
        ss_between += len(group) * (group_mean - grand_mean) ** 2

    # Total sum of squares
    ss_total = ((num_series - grand_mean) ** 2).sum()

    # Eta coefficient
    if ss_total == 0:
        eta = 0.0
    else:
        eta = float(np.sqrt(ss_between / ss_total))

    # Interpretation
    if eta < 0.1:
        strength = "negligible"
    elif eta < 0.3:
        strength = "weak"
    elif eta < 0.5:
        strength = "moderate"
    elif eta < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    # Calculate ANOVA p-value
    groups = [num_series[cat_series == cat].values for cat in cat_series.unique()]
    if len(groups) > 1:
        _, p_val = f_oneway(*groups)
    else:
        p_val = 1.0

    return eta, strength, p_val


def plot_distribution_by_group(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    mode: str = "box",
    top_k: int | None = None,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Plot numeric distribution grouped by categorical column in multiple modes.

    Always calculates and displays eta coefficient (correlation ratio) which measures
    the strength of association between the categorical grouping and numeric distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cat_col : str
        Categorical column for grouping.
    num_col : str
        Numeric column to visualize.
    mode : str, optional
        Visualization mode: "box" (box plot), "violin" (violin plot), "hist" (histogram).
        Default: "box".
    top_k : int, optional
        Limit to top K categories by count. None uses EDAConfig default.
    low_trim : float, optional
        Lower quantile clip for histogram mode. Default: 0.0.
    high_trim : float, optional
        Upper quantile clip for histogram mode. Default: 1.0.
    show_stats : bool, optional
        If True, annotate with n, mean, std for each group. Default: True.
    cfg : EDAConfig, optional
        Visualization config.

    Returns
    -------
    go.Figure
        Plotly figure with eta coefficient and strength interpretation in title.

    Examples
    --------
    >>> fig = plot_distribution_by_group(df, cat_col="education", num_col="age", mode="violin")
    >>> fig.show()
    """
    require_columns(df, [cat_col, num_col])
    require_numeric(df[num_col])
    c = _cfg(cfg)

    data = df[[cat_col, num_col]].dropna()
    cats = data[cat_col].value_counts().head(top_k or c.top_k_categories).index
    data = data[data[cat_col].isin(cats)]

    # Calculate eta coefficient (explains how much numeric variance is explained by categories)
    eta, eta_strength, p_val = _calculate_eta(data[cat_col], data[num_col])
    p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
    sig = "significant" if p_val < 0.05 else "NOT significant"
    title_base = f"{num_col} by {cat_col} | η={eta:.3f} ({p_str}) | {eta_strength}, {sig}"

    fig = go.Figure()

    if mode == "box":
        for i, cat in enumerate(cats):
            sub = data[data[cat_col] == cat][num_col]
            fig.add_trace(
                go.Box(
                    y=sub,
                    name=str(cat),
                    marker_color=c.color_palette[i % len(c.color_palette)],
                )
            )
        fig.update_layout(
            title=title_base,
            yaxis_title=num_col,
            xaxis_title=cat_col,
            showlegend=False,
        )

    elif mode == "violin":
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
            title=title_base,
            yaxis_title=num_col,
            xaxis_title=cat_col,
            showlegend=False,
        )

    elif mode == "hist":
        for i, grp in enumerate(sorted(data[cat_col].unique())):
            s = clip_quantiles(data[data[cat_col] == grp][num_col], low_trim, high_trim)
            fig.add_trace(
                go.Histogram(
                    x=s,
                    nbinsx=30,
                    name=str(grp),
                    histnorm="probability density",
                    marker_color=c.color_palette[i % len(c.color_palette)],
                    opacity=0.65,
                )
            )
        fig.update_layout(
            barmode="overlay",
            title=title_base,
            xaxis_title=num_col,
            yaxis_title="Density",
        )

    fig.update_layout(
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Categorical vs Categorical
# ---------------------------------------------------------------------------


def _interpret_cramers_v(v: float, p_val: float) -> str:
    """Interpret Cramér's V strength and significance.

    Parameters
    ----------
    v : float
        Cramér's V coefficient (0 to 1)
    p_val : float
        P-value from chi-square test

    Returns
    -------
    str
        Interpretation text: "strength, significance"
    """
    # Strength
    if v < 0.1:
        strength = "negligible"
    elif v < 0.3:
        strength = "weak"
    elif v < 0.5:
        strength = "moderate"
    elif v < 0.7:
        strength = "strong"
    else:
        strength = "very strong"

    # Significance
    sig = "significant" if p_val < 0.05 else "NOT significant"

    return f"{strength}, {sig}"


def plot_category_heatmap(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    normalize: bool = True,
    top_k1: int | None = None,
    top_k2: int | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Heatmap of categorical cross-tabulation with Cramér's V association metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col1 : str
        First categorical column (rows).
    col2 : str
        Second categorical column (columns).
    normalize : bool, optional
        If True, show row-percentage; if False, show counts. Default: True.
    top_k1 : int, optional
        Limit col1 to top K categories by count. None uses EDAConfig default.
    top_k2 : int, optional
        Limit col2 to top K categories by count. None uses EDAConfig default.
    cfg : EDAConfig, optional
        Visualization config.

    Returns
    -------
    go.Figure
        Plotly heatmap figure.

    Examples
    --------
    >>> fig = plot_category_heatmap(df, col1="education", col2="income", normalize=True)
    >>> fig.show()
    """
    require_columns(df, [col1, col2])
    c = _cfg(cfg)

    data = df[[col1, col2]].dropna()
    if top_k1:
        data[col1] = top_k_series(data[col1], top_k1)
    if top_k2:
        data[col2] = top_k_series(data[col2], top_k2)

    ct = pd.crosstab(data[col1], data[col2], normalize="index" if normalize else False)

    # Compute Cramér's V
    chi2, p_val, _, _ = chi2_contingency(pd.crosstab(data[col1], data[col2]))
    n = len(data)
    k = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if (n > 0 and k > 0) else 0.0
    v_interp = _interpret_cramers_v(cramers_v, p_val)
    p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"

    fig = go.Figure(
        data=go.Heatmap(
            z=ct.values * (100 if normalize else 1),
            x=ct.columns.astype(str),
            y=ct.index.astype(str),
            colorscale="Blues",
            text=np.round(ct.values * (100 if normalize else 1), 1),
            texttemplate="%{text:.1f}" if normalize else "%{text:.0f}",
            textfont={"size": 10},
            hovertemplate="%{y} × %{x}: %{text:.1f}%<extra></extra>"
            if normalize
            else "%{y} × %{x}: %{text:.0f}<extra></extra>",
        )
    )

    title = f"{col1} vs {col2} | V={cramers_v:.3f} ({p_str}) | {v_interp}"
    fig.update_layout(
        title=title,
        xaxis_title=col2,
        yaxis_title=col1,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Time Series
# ---------------------------------------------------------------------------


def plot_time_series(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    group_by: str | None = None,
    ma: int = 0,
    band: str | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Time series plot with optional moving average, multi-series, and uncertainty bands.

    Always displays trend correlation (r value, p-value) and strength/significance in the title.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (should be sorted by x if x is temporal).
    x : str
        Column for x-axis (typically time or sequential index).
    y : str | list[str]
        Column name(s) for y-axis. If list, plot multiple series.
    group_by : str, optional
        Column to split series by. Each group gets separate color/trace.
    ma : int, optional
        Moving average window. 0 = no MA. Default: 0.
    band : str, optional
        Uncertainty band mode: "std" (mean ± std) or "minmax" (min/max range).
        Only works when group_by is set. Default: None.
    cfg : EDAConfig, optional
        Visualization config.

    Returns
    -------
    go.Figure
        Plotly time series figure with trend correlation annotation.

    Examples
    --------
    >>> fig = plot_time_series(df, x="date", y="revenue", group_by="region", ma=7)
    >>> fig.show()
    """
    c = _cfg(cfg)

    # Ensure y is a list
    y_cols = y if isinstance(y, list) else [y]

    require_columns(df, [x] + y_cols + ([group_by] if group_by else []))
    require_numeric(df[x])
    for y_col in y_cols:
        require_numeric(df[y_col])

    # Sort by x
    data = df[[x] + y_cols + ([group_by] if group_by else [])].dropna().sort_values(x)

    # Calculate trend correlation (r, p-value) for first y column across all data
    corr_text = ""
    if len(data) > 2:
        x_numeric = np.arange(len(data))  # 0, 1, 2, ...
        y_vals = data[y_cols[0]].values
        r, p_val = pearsonr(x_numeric, y_vals)
        p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
        interp_text = _interpret_correlation(r, p_val)
        corr_text = f" | r={r:.3f} ({p_str}) | {interp_text}"

    fig = go.Figure()

    if group_by:
        groups = sorted(data[group_by].unique())
        for g_idx, grp in enumerate(groups):
            sub = data[data[group_by] == grp]

            for y_idx, y_col in enumerate(y_cols):
                color = c.color_palette[(g_idx + y_idx) % len(c.color_palette)]
                trace_name = f"{y_col} ({grp})" if len(y_cols) > 1 else str(grp)

                fig.add_trace(
                    go.Scatter(
                        x=sub[x],
                        y=sub[y_col],
                        mode="lines",
                        name=trace_name,
                        line={"color": color, "width": 1.5},
                    )
                )

                # Add MA if requested
                if ma > 0:
                    ma_vals = sub[y_col].rolling(ma, min_periods=1).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x],
                            y=ma_vals,
                            mode="lines",
                            name=f"{trace_name} MA({ma})",
                            line={"color": color, "width": 2, "dash": "dash"},
                            showlegend=True,
                        )
                    )

                # Add uncertainty band
                if band == "std":
                    y_mean = sub[y_col].rolling(ma or 30, min_periods=1).mean()
                    y_std = sub[y_col].rolling(ma or 30, min_periods=1).std()
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x],
                            y=y_mean + y_std,
                            mode="lines",
                            line={"width": 0},
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x],
                            y=y_mean - y_std,
                            mode="lines",
                            line={"width": 0},
                            fillcolor=color,
                            fill="tonexty",
                            name=f"{trace_name} ±std",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
    else:
        for y_idx, y_col in enumerate(y_cols):
            color = c.color_palette[y_idx % len(c.color_palette)]

            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[y_col],
                    mode="lines",
                    name=y_col,
                    line={"color": color, "width": 1.5},
                )
            )

            # Add MA if requested
            if ma > 0:
                ma_vals = data[y_col].rolling(ma, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data[x],
                        y=ma_vals,
                        mode="lines",
                        name=f"{y_col} MA({ma})",
                        line={"color": color, "width": 2, "dash": "dash"},
                    )
                )

    y_title = "/".join(y_cols) if len(y_cols) > 1 else y_cols[0]
    fig.update_layout(
        title=f"{y_title} over {x}{corr_text}",
        xaxis_title=x,
        yaxis_title=y_title,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


# ---------------------------------------------------------------------------
# Association Ranking Table
# ---------------------------------------------------------------------------


def rank_associations(
    df: pd.DataFrame,
    target: str,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    num_method: str = "pearson",
    unique_threshold: int = 50,
) -> pd.DataFrame:
    """Rank numeric and categorical features by association strength with target.

    Numeric features: Pearson/Spearman correlation with p-value.
    Categorical features: Cramér's V with chi-square p-value.
    All results sorted by |association| descending.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Target column (numeric or categorical).
    num_cols : list[str], optional
        Numeric columns to test. If None, auto-detected.
    cat_cols : list[str], optional
        Categorical columns to test. If None, auto-detected via unique_threshold.
    num_method : str, optional
        Correlation method for numeric: "pearson", "spearman", "kendall". Default: "pearson".
    unique_threshold : int, optional
        Max unique values to consider column categorical when auto-detecting. Default: 50.

    Returns
    -------
    pd.DataFrame
        Columns: feature, association, p_value, method.
        Sorted by |association| descending.

    Examples
    --------
    >>> ranking = rank_associations(df, target="income")
    >>> ranking.head()
    """
    require_column(df, target)

    # Auto-detect columns if not provided
    if num_cols is None:
        num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols is None:
        cat_cols = [c for c in df.columns if c != target and c not in num_cols and df[c].nunique() <= unique_threshold]

    results = []

    # Numeric associations
    if pd.api.types.is_numeric_dtype(df[target]):
        target_numeric = df[target]
        for col in num_cols:
            mask = target_numeric.notna() & df[col].notna()
            if mask.sum() < 2:
                continue

            if num_method == "pearson":
                assoc, p_val = pearsonr(target_numeric[mask], df[col][mask])
            elif num_method == "spearman":
                assoc, p_val = spearmanr(target_numeric[mask], df[col][mask])
            else:
                assoc, p_val = spearmanr(target_numeric[mask], df[col][mask])

            results.append(
                {
                    "feature": col,
                    "association": round(assoc, 4),
                    "p_value": round(p_val, 4),
                    "method": num_method,
                }
            )

    # Categorical associations
    for col in cat_cols:
        s1 = df[col].astype(str)
        s2 = df[target].astype(str)
        mask = s1.notna() & s2.notna()
        if mask.sum() < 2:
            continue

        # Cramér's V
        ct = pd.crosstab(s1[mask], s2[mask])
        chi2, p_val, _, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        if n > 0 and k > 0:
            cramers_v_val = np.sqrt(chi2 / (n * k))
        else:
            cramers_v_val = 0.0

        results.append(
            {
                "feature": col,
                "association": round(cramers_v_val, 4),
                "p_value": round(p_val, 4),
                "method": "cramers_v",
            }
        )

    result_df = pd.DataFrame(results)
    if len(result_df) > 0:
        result_df = result_df.reindex(result_df["association"].abs().argsort()[::-1]).reset_index(drop=True)

    return result_df


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------


def plot_roc_curve(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """ROC curve with AUC annotation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    label_col : str
        Binary (0/1) label column.
    pred_col : str
        Predicted probability column.
    cfg : EDAConfig, optional
        Visualization config.

    Returns
    -------
    go.Figure
        Plotly ROC curve figure.

    Examples
    --------
    >>> fig = plot_roc_curve(df, label_col="target", pred_col="pred_proba")
    >>> fig.show()
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
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC={auc_val:.4f})",
            line={"color": c.color_palette[0], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line={"color": "gray", "dash": "dash"},
        )
    )

    fig.update_layout(
        title=f"ROC Curve — AUC = {auc_val:.4f}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig
