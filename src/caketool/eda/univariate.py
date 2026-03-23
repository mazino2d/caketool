"""Single-variable (univariate) analysis and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._validators import clip_quantiles, require_nonempty, require_numeric, top_k_series
from .config import EDAConfig


def _cfg(cfg: EDAConfig | None) -> EDAConfig:
    return cfg if cfg is not None else EDAConfig()


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


def histogram(
    series: pd.Series,
    nbins: int = 30,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Plot a histogram of a numeric series.

    Parameters
    ----------
    series : pd.Series
        Numeric series to plot.
    nbins : int
        Number of bins.
    low_trim : float
        Lower quantile to clip outliers (0.0–1.0).
    high_trim : float
        Upper quantile to clip outliers (0.0–1.0).
    cfg : EDAConfig, optional
        Visual configuration.

    Returns
    -------
    go.Figure
    """
    require_numeric(series)
    require_nonempty(series)
    c = _cfg(cfg)
    s = clip_quantiles(series.dropna(), low_trim, high_trim)
    fig = go.Figure(go.Histogram(x=s, nbinsx=nbins, marker_color=c.color_palette[0], opacity=0.85))
    fig.update_layout(
        title=f"Distribution: {series.name}",
        xaxis_title=series.name,
        yaxis_title="Count",
        template=c.template,
        width=c.width,
        height=c.height,
        bargap=0.05,
    )
    return fig


def overlay_histogram(
    series_dict: dict[str, pd.Series],
    nbins: int = 30,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Overlay histograms of multiple numeric series.

    Parameters
    ----------
    series_dict : dict[str, pd.Series]
        Mapping of label → series.
    nbins : int
        Number of bins.
    low_trim : float
        Lower quantile clip.
    high_trim : float
        Upper quantile clip.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    c = _cfg(cfg)
    fig = go.Figure()
    for i, (label, s) in enumerate(series_dict.items()):
        require_numeric(s)
        require_nonempty(s)
        clipped = clip_quantiles(s.dropna(), low_trim, high_trim)
        fig.add_trace(
            go.Histogram(
                x=clipped,
                nbinsx=nbins,
                name=label,
                marker_color=c.color_palette[i % len(c.color_palette)],
                opacity=0.65,
            )
        )
    first_name = next(iter(series_dict))
    fig.update_layout(
        barmode="overlay",
        title=f"Overlay Distribution: {first_name}",
        xaxis_title=first_name,
        yaxis_title="Count",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def distribution(
    series: pd.Series,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Plot a KDE distribution curve (via histogram + kde overlay).

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    low_trim : float
        Lower quantile clip.
    high_trim : float
        Upper quantile clip.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    from scipy.stats import gaussian_kde

    require_numeric(series)
    require_nonempty(series)
    c = _cfg(cfg)
    s = clip_quantiles(series.dropna(), low_trim, high_trim).values.astype(float)
    kde = gaussian_kde(s)
    x_range = np.linspace(s.min(), s.max(), 300)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=s,
            nbinsx=40,
            histnorm="probability density",
            marker_color=c.color_palette[0],
            opacity=0.5,
            name="Histogram",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=kde(x_range),
            mode="lines",
            line={"color": c.color_palette[1], "width": 2},
            name="KDE",
        )
    )
    fig.update_layout(
        title=f"Distribution (KDE): {series.name}",
        xaxis_title=series.name,
        yaxis_title="Density",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def percentile_table(
    series: pd.Series,
    step: int = 5,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
) -> pd.DataFrame:
    """Compute a percentile summary table.

    Parameters
    ----------
    series : pd.Series
        Numeric series.
    step : int
        Percentile step size (e.g. 5 → 0%, 5%, 10%, ..., 100%).
    low_trim : float
        Lower quantile clip before computing.
    high_trim : float
        Upper quantile clip before computing.

    Returns
    -------
    pd.DataFrame
        Columns: percentile, value, count_below.
    """
    require_numeric(series)
    require_nonempty(series)
    s = clip_quantiles(series.dropna(), low_trim, high_trim)
    # Always include fine-grained extremes
    pcts = sorted(set([0.01, 0.1, 0.5, 1.0] + list(range(0, 101, step)) + [99.0, 99.5, 99.9, 99.99]))
    pcts = [p for p in pcts if 0 <= p <= 100]
    values = [s.quantile(p / 100) for p in pcts]
    result = pd.DataFrame({"percentile": pcts, f"{series.name}_value": values})
    result["percentile"] = result["percentile"].map(lambda x: f"{x:.2f}%")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------


def value_counts(
    series: pd.Series,
    top_k: int = 20,
    dropna: bool = True,
) -> pd.DataFrame:
    """Compute value counts with frequency percentage.

    Parameters
    ----------
    series : pd.Series
        Any series.
    top_k : int
        Limit to top *k* most frequent values.
    dropna : bool
        Exclude NaN from counts.

    Returns
    -------
    pd.DataFrame
        Columns: value, count, pct.
    """
    require_nonempty(series)
    vc = series.value_counts(dropna=dropna).head(top_k)
    total = series.notna().sum() if dropna else len(series)
    df = pd.DataFrame({"value": vc.index, "count": vc.values})
    df["pct"] = (df["count"] / total * 100).round(2)
    return df.reset_index(drop=True)


def bar_count(
    series: pd.Series,
    top_k: int = 20,
    dropna: bool = True,
    horizontal: bool = False,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Bar chart of value frequencies.

    Parameters
    ----------
    series : pd.Series
        Categorical series.
    top_k : int
        Top K categories to display.
    dropna : bool
        Exclude NaN.
    horizontal : bool
        Flip axes.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_nonempty(series)
    c = _cfg(cfg)
    df = value_counts(series, top_k=top_k, dropna=dropna)
    labels = df["value"].astype(str)
    counts = df["count"]
    if horizontal:
        trace = go.Bar(y=labels, x=counts, orientation="h", marker_color=c.color_palette[0])
        xaxis, yaxis = {"title": "Count"}, {"title": series.name, "autorange": "reversed"}
    else:
        trace = go.Bar(x=labels, y=counts, marker_color=c.color_palette[0])
        xaxis, yaxis = {"title": series.name}, {"title": "Count"}
    fig = go.Figure(trace)
    fig.update_layout(
        title=f"Value Counts: {series.name}",
        xaxis=xaxis,
        yaxis=yaxis,
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def pie_chart(
    series: pd.Series,
    top_k: int = 8,
    dropna: bool = True,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Pie chart with "Others" grouping for low-frequency categories.

    Parameters
    ----------
    series : pd.Series
        Categorical series.
    top_k : int
        Top categories to show individually.
    dropna : bool
        Exclude NaN.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_nonempty(series)
    c = _cfg(cfg)
    s = series.dropna() if dropna else series
    s = top_k_series(s, k=top_k)
    vc = s.value_counts()
    fig = go.Figure(
        go.Pie(
            labels=vc.index.astype(str),
            values=vc.values,
            marker={"colors": c.color_palette},
            hole=0.3,
        )
    )
    fig.update_layout(
        title=f"Distribution: {series.name}",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig
