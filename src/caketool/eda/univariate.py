"""Single-variable (univariate) analysis and visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._validators import clip_quantiles, require_nonempty, require_numeric
from .config import EDAConfig


def _cfg(cfg: EDAConfig | None) -> EDAConfig:
    return cfg if cfg is not None else EDAConfig()


_STAT_LINES: list[tuple[str, str, str]] = [
    # (label, dash_style, color)
    ("Lower Fence", "dot", "#D35400"),
    ("Q1", "dash", "#2471A3"),
    ("Median", "solid", "#C0392B"),
    ("Mean", "dashdot", "#1E8449"),
    ("Q3", "dash", "#7D3C98"),
    ("Upper Fence", "dot", "#D35400"),
]


def _add_stat_info(fig: go.Figure, s: pd.Series, c: EDAConfig) -> None:
    """Add Q1/Median/Q3/Mean/Fence vertical reference lines and a stats table to *fig*."""
    q1 = float(s.quantile(0.25))
    median = float(s.quantile(0.50))
    q3 = float(s.quantile(0.75))
    mean = float(s.mean())
    iqr = q3 - q1
    lower_fence = max(q1 - 1.5 * iqr, float(s.min()))
    upper_fence = min(q3 + 1.5 * iqr, float(s.max()))

    stats = {
        "Lower Fence": lower_fence,
        "Q1": q1,
        "Median": median,
        "Mean": mean,
        "Q3": q3,
        "Upper Fence": upper_fence,
    }

    # Draw vertical lines
    for label, dash, color in _STAT_LINES:
        fig.add_shape(
            type="line",
            x0=stats[label],
            x1=stats[label],
            y0=0,
            y1=1,
            yref="paper",
            line={"color": color, "width": 2.5, "dash": dash},
        )

    # Build a single stats table as HTML annotation
    _dash_symbols = {"solid": "\u2500", "dash": "- -", "dashdot": "-\u00b7-", "dot": "\u00b7\u00b7\u00b7"}
    rows = []
    for label, dash, color in _STAT_LINES:
        symbol = _dash_symbols.get(dash, "\u2500")
        rows.append(f'<span style="color:{color}"><b>{symbol}</b></span> {label}: <b>{stats[label]:,.2f}</b>')
    table_text = "<br>".join(rows)

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        text=table_text,
        showarrow=False,
        font={"size": 11, "family": "monospace"},
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#cccccc",
        borderwidth=1,
        borderpad=8,
    )


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


def plot_distribution(
    data: pd.Series | dict[str, pd.Series],
    nbins: int = 40,
    kde: bool = False,
    low_trim: float = 0.0,
    high_trim: float = 1.0,
    show_stats: bool = True,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Plot distribution of one or more numeric series.

    Unified function that handles single histogram, overlay histogram,
    KDE distribution, and overlay KDE distribution.

    * Single series → histogram (optionally with KDE overlay).
    * Multiple series (dict) → overlaid histograms (optionally with KDE).
    * ``show_stats`` adds Q1/Median/Q3/Mean/Fence lines (single series only).

    Parameters
    ----------
    data : pd.Series | dict[str, pd.Series]
        A single numeric series **or** a ``{label: series}`` dict for overlay.
    nbins : int
        Number of histogram bins.
    kde : bool
        If True, overlay a KDE smoothing curve on each series.
    low_trim : float
        Lower quantile to clip outliers (0.0–1.0).
    high_trim : float
        Upper quantile to clip outliers (0.0–1.0).
    show_stats : bool
        If True and *data* is a single series, draw vertical reference lines
        for Q1, Median, Q3, Mean, Lower Fence and Upper Fence (Tukey 1.5×IQR).
    cfg : EDAConfig, optional
        Visual configuration.

    Returns
    -------
    go.Figure
    """
    if isinstance(data, pd.Series):
        series_dict = {str(data.name): data}
        is_single = True
    else:
        series_dict = data
        is_single = len(series_dict) == 1

    c = _cfg(cfg)
    fig = go.Figure()
    clipped_first: pd.Series | None = None

    for i, (label, s) in enumerate(series_dict.items()):
        require_numeric(s)
        require_nonempty(s)
        clipped = clip_quantiles(s.dropna(), low_trim, high_trim)
        if clipped_first is None:
            clipped_first = clipped
        color = c.color_palette[i % len(c.color_palette)]
        is_overlay = not is_single

        hist_opacity = 0.5 if kde and is_single else 0.35 if kde else 0.65 if is_overlay else 0.85
        fig.add_trace(
            go.Histogram(
                x=clipped,
                nbinsx=nbins,
                name="Histogram" if is_single else label,
                marker_color=color,
                opacity=hist_opacity,
                showlegend=not (kde and is_overlay),
            )
        )

        if kde:
            from scipy.stats import gaussian_kde

            arr = clipped.values.astype(float)
            kde_fn = gaussian_kde(arr)
            n = len(arr)
            bin_width = (arr.max() - arr.min()) / nbins
            x_range = np.linspace(arr.min(), arr.max(), 300)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=kde_fn(x_range) * n * bin_width,
                    mode="lines",
                    name="KDE" if is_single else label,
                    line={"color": c.color_palette[1] if is_single else color, "width": 2},
                )
            )

    first_name = next(iter(series_dict))
    title_prefix = "Distribution"
    if not is_single:
        title_prefix = "Overlay Distribution"
    if kde:
        title_prefix += " (KDE)"

    fig.update_layout(
        barmode="overlay" if not is_single else None,
        title=f"{title_prefix}: {first_name}",
        xaxis_title=first_name,
        yaxis_title="Count",
        template=c.template,
        width=c.width,
        height=c.height,
        bargap=0.05 if is_single else 0,
    )

    if show_stats and is_single and clipped_first is not None:
        _add_stat_info(fig, clipped_first, c)

    return fig


def compute_quantile(
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


def compute_frequency(
    series: pd.Series,
    top_k: int = 20,
    dropna: bool = False,
) -> pd.DataFrame:
    """Compute value counts with frequency percentage.

    Returns the *top_k* most frequent non-null values, an ``Others`` row
    aggregating remaining categories (if any), and a ``NaN`` row when
    ``dropna=False`` and missing values exist.

    Parameters
    ----------
    series : pd.Series
        Any series.
    top_k : int
        Limit to top *k* most frequent non-null values.
    dropna : bool
        If False, include a NaN row in the result when missing values exist.

    Returns
    -------
    pd.DataFrame
        Columns: value, count, pct.
    """
    require_nonempty(series)
    total = len(series)
    nan_count = int(series.isna().sum())

    # Non-null value counts
    vc = series.value_counts(dropna=True)
    top_vc = vc.head(top_k)
    others_count = int(vc.iloc[top_k:].sum()) if len(vc) > top_k else 0

    rows: list[dict] = [{"value": val, "count": cnt} for val, cnt in zip(top_vc.index, top_vc.values, strict=True)]
    if others_count > 0:
        rows.append({"value": "Others", "count": others_count})
    if not dropna and nan_count > 0:
        rows.append({"value": "NaN", "count": nan_count})

    df = pd.DataFrame(rows)
    df["pct"] = (df["count"] / total * 100).round(2)
    return df.reset_index(drop=True)


def plot_frequency(
    series: pd.Series,
    top_k: int = 15,
    dropna: bool = False,
    mode: str = "pie",
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Visualize categorical frequency as bar chart or pie chart.

    Parameters
    ----------
    series : pd.Series
        Categorical series.
    top_k : int
        Top K categories to display individually; remaining are grouped
        into an "Others" slice/bar.
    dropna : bool
        If False, include a NaN bar/slice when missing values exist.
    mode : str
        One of ``"pie"`` (default), ``"bar"`` (vertical bar), or
        ``"barh"`` (horizontal bar).
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_nonempty(series)
    c = _cfg(cfg)
    df = compute_frequency(series, top_k=top_k, dropna=dropna)
    labels = df["value"].astype(str)
    counts = df["count"]

    if mode == "pie":
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=counts,
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
    elif mode in ("bar", "barh"):
        horizontal = mode == "barh"
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
    else:
        msg = f"mode must be 'pie', 'bar', or 'barh', got {mode!r}"
        raise ValueError(msg)

    return fig
