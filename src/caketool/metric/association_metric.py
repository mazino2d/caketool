"""Pairwise association metrics for numeric and categorical variables."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr


def association(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: Literal["pearson", "spearman", "eta", "cramers_v"],
) -> tuple[float, float]:
    """Compute pairwise association between two variables.

    Selects the appropriate statistical metric based on ``method`` and returns
    a ``(value, p_value)`` tuple.

    Parameters
    ----------
    x : pd.Series | np.ndarray
        First variable. For ``"eta"``, this must be the categorical variable.
    y : pd.Series | np.ndarray
        Second variable. For ``"eta"``, this must be the numeric variable.
    method : {"pearson", "spearman", "eta", "cramers_v"}
        Association metric to compute:

        - ``"pearson"``   – Pearson correlation for two numeric variables.
          Returns signed *r* in [-1, 1].
        - ``"spearman"``  – Spearman rank correlation for two numeric variables.
          Returns signed *ρ* in [-1, 1].
        - ``"eta"``       – Eta coefficient (correlation ratio) for categorical
          ``x`` and numeric ``y``. Returns *η* in [0, 1].
        - ``"cramers_v"`` – Cramér's V for two categorical variables.
          Returns *V* in [0, 1].

    Returns
    -------
    tuple[float, float]
        ``(association_value, p_value)``

    Notes
    -----
    ``x`` and ``y`` are paired by position, not by index, and pairs where
    either value is missing are dropped before computing the metric.

    Examples
    --------
    >>> association(df["age"], df["income"], method="pearson")
    (0.45, 0.001)
    >>> association(df["education"], df["age"], method="eta")
    (0.31, 0.0001)
    >>> association(df["education"], df["marital"], method="cramers_v")
    (0.22, 0.0)
    """
    if method not in ("pearson", "spearman", "eta", "cramers_v"):
        raise ValueError(f"method must be one of 'pearson', 'spearman', 'eta', 'cramers_v'. Got: {method!r}")

    # Align by position (not index) and drop pairs where either side is missing
    x_s = pd.Series(x).reset_index(drop=True)
    y_s = pd.Series(y).reset_index(drop=True)
    if len(x_s) != len(y_s):
        raise ValueError(f"x and y must have the same length. Got: {len(x_s)} and {len(y_s)}")
    mask = x_s.notna() & y_s.notna()
    x_s, y_s = x_s[mask].reset_index(drop=True), y_s[mask].reset_index(drop=True)

    if method in ("pearson", "spearman"):
        x_arr = x_s.to_numpy(dtype=float)
        y_arr = y_s.to_numpy(dtype=float)
        if method == "pearson":
            stat, p_val = pearsonr(x_arr, y_arr)
        else:
            stat, p_val = spearmanr(x_arr, y_arr)
        return float(stat), float(p_val)

    if method == "eta":
        cat = x_s
        num = y_s.astype(float)
        grand_mean = num.mean()
        ss_between = sum(len(num[cat == c]) * (num[cat == c].mean() - grand_mean) ** 2 for c in cat.unique())
        ss_total = float(((num - grand_mean) ** 2).sum())
        eta = float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0
        groups = [num[cat == c].to_numpy() for c in cat.unique()]
        _, p_val = f_oneway(*groups) if len(groups) > 1 else (None, 1.0)
        return eta, float(p_val)

    ct = pd.crosstab(x_s.astype(str), y_s.astype(str))
    if min(ct.shape) < 2:
        return 0.0, 1.0
    # No Yates continuity correction: it biases V downward on 2x2 tables
    chi2, p_val, _, _ = chi2_contingency(ct, correction=False)
    n = int(ct.values.sum())
    k = min(ct.shape) - 1
    v = float(np.sqrt(chi2 / (n * k))) if (n > 0 and k > 0) else 0.0
    return v, float(p_val)
