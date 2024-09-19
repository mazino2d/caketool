from typing import List

import pandas as pd


def decribe_risk_score(
    score_df: pd.DataFrame, 
    pred_col: str = "score", 
    label_col: str = "label",
    segments: List[float] = [
        0, .01, .02, .03, .04, .05, .06, .07, .08, .09,
        .1, .125, .15, .2, .3, .4, .5, .6, .8, .9, 1
    ]
) -> pd.DataFrame:
    prop_df = score_df[[pred_col, label_col]].copy()
    prop_df.columns = ["probability", "default_next_month"]
    prop_df['proba_segment'] = pd.cut(prop_df['probability'], segments, include_lowest=True)
    summary = prop_df.groupby('proba_segment', observed=False).agg(
        client=("default_next_month", "count"),
        bad=("default_next_month", lambda s: (s == 1).sum()),
        good=("default_next_month", lambda s: (s == 0).sum()),
    )
    summary["client_cumsum"] = summary["client"].cumsum()
    summary["bad_cumsum"] = summary["bad"].cumsum()
    summary['good_cumsum'] = summary['good'].cumsum()
    summary["%bad"] = round((summary["bad"]/summary["bad_cumsum"].max()) * 100, 2)
    summary["%bad_cumsum"] = summary["%bad"].cumsum()
    summary["%good"] = round(summary["good"]/summary["good_cumsum"].max() * 100, 2)
    summary["%good_cumsum"] = summary["%good"].cumsum()
    summary["def_rate%"] = round((summary['bad_cumsum'] / summary['client_cumsum']) * 100, 2)
    summary["approval_rate%"] = round(summary["client_cumsum"] / len(prop_df) * 100, 2)
    summary = summary.reset_index()
    summary["band"] = [f"B{i}" for i in range(len(summary), 0, -1)]

    return summary[[
        'band', 'proba_segment', 
        'def_rate%', 'approval_rate%',
        '%bad_cumsum', '%good_cumsum',
        '%bad', '%good',
        'client_cumsum', 'bad_cumsum', 'good_cumsum',
        'client', 'bad', 'good',
    ]]
