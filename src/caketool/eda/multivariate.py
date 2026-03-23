"""Multi-variable (3D+) analysis and visualization."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ._validators import require_columns, top_k_series
from .config import EDAConfig


def _cfg(cfg: EDAConfig | None) -> EDAConfig:
    return cfg if cfg is not None else EDAConfig()


def parallel_coordinates(
    df: pd.DataFrame,
    dims: list[str],
    color_by: str | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Parallel coordinates plot for multi-dimensional exploration.

    Parameters
    ----------
    df : pd.DataFrame
    dims : list[str]
        Numeric columns to include as dimensions.
    color_by : str, optional
        Numeric column to encode as color.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, dims + ([color_by] if color_by else []))
    c = _cfg(cfg)
    all_cols = list(dict.fromkeys(dims + ([color_by] if color_by else [])))
    data = df[all_cols].dropna()
    dimensions = [{"label": col, "values": data[col], "range": (data[col].min(), data[col].max())} for col in dims]
    line_kw: dict = {"color": c.color_palette[0]}
    if color_by:
        line_kw = {"color": data[color_by], "colorscale": "Viridis", "showscale": True, "colorbar": {"title": color_by}}
    fig = go.Figure(go.Parcoords(line=line_kw, dimensions=dimensions))
    fig.update_layout(
        title="Parallel Coordinates",
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def scatter_3d(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    color_by: str | None = None,
    sample_n: int | None = 3000,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """3D scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
    x, y, z : str
        Numeric column names for axes.
    color_by : str, optional
        Column for color grouping.
    sample_n : int, optional
        Random sample size. None = all rows.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [x, y, z] + ([color_by] if color_by else []))
    c = _cfg(cfg)
    cols = [x, y, z] + ([color_by] if color_by else [])
    data = df[cols].dropna()
    if sample_n and len(data) > sample_n:
        data = data.sample(n=sample_n, random_state=42)
    fig = go.Figure()
    if color_by:
        for i, grp in enumerate(sorted(data[color_by].unique())):
            sub = data[data[color_by] == grp]
            fig.add_trace(
                go.Scatter3d(
                    x=sub[x],
                    y=sub[y],
                    z=sub[z],
                    mode="markers",
                    name=str(grp),
                    marker={"size": 3, "color": c.color_palette[i % len(c.color_palette)], "opacity": 0.7},
                )
            )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=data[x],
                y=data[y],
                z=data[z],
                mode="markers",
                marker={"size": 3, "color": c.color_palette[0], "opacity": 0.7},
            )
        )
    fig.update_layout(
        title=f"3D Scatter: {x} / {y} / {z}",
        scene={"xaxis_title": x, "yaxis_title": y, "zaxis_title": z},
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig


def scatter_matrix(
    df: pd.DataFrame,
    columns: list[str],
    color_by: str | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Scatter plot matrix (replaces seaborn pairplot).

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Numeric columns to include.
    color_by : str, optional
        Categorical column for color grouping.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, columns + ([color_by] if color_by else []))
    c = _cfg(cfg)
    cols = columns + ([color_by] if color_by else [])
    data = df[cols].dropna()
    dims = [{"label": col, "values": data[col]} for col in columns]
    marker_kw: dict = {"size": 3, "opacity": 0.6, "color": c.color_palette[0]}
    if color_by:
        codes, _ = pd.factorize(data[color_by])
        marker_kw = {
            "size": 3,
            "opacity": 0.6,
            "color": codes,
            "colorscale": "Viridis",
            "showscale": False,
        }
    fig = go.Figure(go.Splom(dimensions=dims, marker=marker_kw, diagonal_visible=True))
    fig.update_layout(
        title="Scatter Matrix",
        template=c.template,
        width=max(c.width, 800),
        height=max(c.height, 800),
    )
    return fig


def stacked_bar(
    df: pd.DataFrame,
    x: str,
    category: str,
    value: str | None = None,
    normalize: bool = False,
    top_k: int | None = None,
    cfg: EDAConfig | None = None,
) -> go.Figure:
    """Stacked bar chart.

    Parameters
    ----------
    df : pd.DataFrame
    x : str
        x-axis grouping column.
    category : str
        Column to stack by.
    value : str, optional
        Numeric value column to aggregate (sum). If None, count rows.
    normalize : bool
        If True, show percentage (100% stacked).
    top_k : int, optional
        Limit *category* to top K values.
    cfg : EDAConfig, optional

    Returns
    -------
    go.Figure
    """
    require_columns(df, [x, category] + ([value] if value else []))
    c = _cfg(cfg)
    data = df.copy()
    if top_k:
        data[category] = top_k_series(data[category], top_k)
    if value:
        ct = data.groupby([x, category])[value].sum().unstack(fill_value=0)
    else:
        ct = data.groupby([x, category]).size().unstack(fill_value=0)
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_trace(
            go.Bar(
                x=ct.index.astype(str),
                y=ct[col],
                name=str(col),
                marker_color=c.color_palette[i % len(c.color_palette)],
            )
        )
    fig.update_layout(
        barmode="stack",
        title=f"{category} composition by {x}",
        xaxis_title=x,
        yaxis_title="Percentage (%)" if normalize else (value or "Count"),
        template=c.template,
        width=c.width,
        height=c.height,
    )
    return fig
