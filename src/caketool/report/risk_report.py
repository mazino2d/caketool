import pandas as pd


def decribe_risk_score(
    score_df: pd.DataFrame,
    pred_col: str = "score",
    label_col: str = "label",
    segments: tuple[float, ...] = (  # noqa: E501
        0,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
        0.125,
        0.15,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        0.9,
        1,
    ),
) -> pd.DataFrame:
    """Segment predicted risk scores into probability bands and compute key credit metrics.

    Cuts the predicted probability column into the provided *segments* and
    aggregates, per band, the number of clients, defaults (bad), and non-defaults
    (good).  Then derives cumulative statistics, default rates, and approval rates
    commonly used in scorecard validation and monitoring.

    Bands are labelled ``B1`` (highest-risk band, probability closest to 1) down
    to ``BN`` (lowest-risk band), following the convention that lower-score
    applicants are rejected first.

    Parameters
    ----------
    score_df : pd.DataFrame
        DataFrame containing at least the prediction and label columns.
    pred_col : str, optional
        Name of the predicted probability column.  Defaults to ``"score"``.
    label_col : str, optional
        Name of the binary target column (1 = default/bad, 0 = good).
        Defaults to ``"label"``.
    segments : tuple[float, ...], optional
        Monotonically increasing probability cut-points in [0, 1] that define
        the band boundaries.  The default provides fine granularity at low
        probabilities (0–10 %) and coarser granularity at higher probabilities,
        which is typical for credit scoring where most applicants have low
        default probability.

    Returns
    -------
    pd.DataFrame
        One row per probability band with the following columns:

        - ``band`` – Band label (``"B1"`` = highest risk … ``"BN"`` = lowest risk).
        - ``proba_segment`` – Pandas ``Interval`` showing the probability range.
        - ``def_rate%`` – Cumulative default rate up to and including this band
          (``bad_cumsum / client_cumsum * 100``).
        - ``approval_rate%`` – Cumulative approval rate up to and including
          this band (``client_cumsum / total_clients * 100``).
        - ``%bad_cumsum`` – Cumulative percentage of all defaults captured.
        - ``%good_cumsum`` – Cumulative percentage of all non-defaults captured.
        - ``%bad`` – Percentage of all defaults in this band.
        - ``%good`` – Percentage of all non-defaults in this band.
        - ``client_cumsum`` – Cumulative client count.
        - ``bad_cumsum`` – Cumulative default count.
        - ``good_cumsum`` – Cumulative non-default count.
        - ``client`` – Client count in this band.
        - ``bad`` – Default count in this band.
        - ``good`` – Non-default count in this band.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"score": [0.05, 0.15, 0.45, 0.85], "label": [0, 1, 0, 1]})
    >>> decribe_risk_score(df, pred_col="score", label_col="label")
    """
    prop_df = score_df[[pred_col, label_col]].copy()
    prop_df.columns = ["probability", "default_next_month"]
    prop_df["proba_segment"] = pd.cut(prop_df["probability"], segments, include_lowest=True)
    summary = prop_df.groupby("proba_segment", observed=False).agg(
        client=("default_next_month", "count"),
        bad=("default_next_month", lambda s: (s == 1).sum()),
        good=("default_next_month", lambda s: (s == 0).sum()),
    )
    summary["client_cumsum"] = summary["client"].cumsum()
    summary["bad_cumsum"] = summary["bad"].cumsum()
    summary["good_cumsum"] = summary["good"].cumsum()
    summary["%bad"] = round((summary["bad"] / summary["bad_cumsum"].max()) * 100, 2)
    summary["%bad_cumsum"] = summary["%bad"].cumsum()
    summary["%good"] = round(summary["good"] / summary["good_cumsum"].max() * 100, 2)
    summary["%good_cumsum"] = summary["%good"].cumsum()
    summary["def_rate%"] = round((summary["bad_cumsum"] / summary["client_cumsum"]) * 100, 2)
    summary["approval_rate%"] = round(summary["client_cumsum"] / len(prop_df) * 100, 2)
    summary = summary.reset_index()
    summary["band"] = [f"B{i}" for i in range(len(summary), 0, -1)]

    return summary[
        [
            "band",
            "proba_segment",
            "def_rate%",
            "approval_rate%",
            "%bad_cumsum",
            "%good_cumsum",
            "%bad",
            "%good",
            "client_cumsum",
            "bad_cumsum",
            "good_cumsum",
            "client",
            "bad",
            "good",
        ]
    ]
