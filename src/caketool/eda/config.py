from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EDAConfig:
    """Global configuration for EDA visualizations.

    Parameters
    ----------
    width : int
        Default figure width in pixels.
    height : int
        Default figure height in pixels.
    template : str
        Plotly template name.
    color_palette : list[str]
        List of hex colors for categorical traces.
    top_k_categories : int
        Max categories to show before grouping as "Others".
    percentile_step : int
        Step size (%) for percentile tables.
    """

    width: int = 900
    height: int = 450
    template: str = "simple_white"
    color_palette: list[str] = field(
        default_factory=lambda: [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
        ]
    )
    top_k_categories: int = 15
    percentile_step: int = 5
