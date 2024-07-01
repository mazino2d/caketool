

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def decribe_risk_score(
    score_df: pd.DataFrame, pred_col: str, label_col: str,
    segments: List[float] = [
        .01, .025, .05, .075, .1, .125, .15,
        .2, .3, .4, .5, .6, .8, .9, 1
    ]
) -> pd.DataFrame:
    segments.sort()

    def get_segment(score: float):
        for s in segments:
            if score <= s:
                return s
        return None

    # creating propensity score based segments
    prop_df = score_df[[pred_col, label_col]]
    prop_df.columns = ["probability", "default_next_month"]
    prop_df['proba_segment'] = prop_df['probability'].apply(get_segment)
    summary = prop_df.groupby('proba_segment').agg(
        client=("default_next_month", "count"),
        bad=("default_next_month", lambda s: (s == 1).sum()),
        good=("default_next_month", lambda s: (s == 0).sum()),
    )
    summary["cumsum_client"] = summary["client"].cumsum()
    summary["cumsum_bad"] = summary["bad"].cumsum()
    summary['cumsum_good'] = summary['good'].cumsum()
    summary["%bad vs total bad"] = round((summary["bad"]/summary["cumsum_bad"].max()) * 100, 2)
    summary["%bad cumsum"] = summary["%bad vs total bad"].cumsum()
    summary["%good vs total good"] = round(summary["good"]/summary["cumsum_good"].max() * 100, 2)
    summary["%good cumsum"] = summary["%good vs total good"].cumsum()
    summary["def_rate%"] = round((summary['cumsum_bad'] / summary['cumsum_client']) * 100, 2)
    summary["approval_rate%"] = round(summary["cumsum_client"] / len(prop_df) * 100, 2)
    summary['FPR'] = 1-summary['cumsum_good'] / (summary['cumsum_good'].max())
    summary['TPR'] = 1-summary['cumsum_bad'] / (summary['cumsum_bad'].max())
    summary[["bad_scaled", "good_scaled"]] = np.round(MinMaxScaler().fit_transform(summary[["bad", "good"]]), 3)
    summary = summary.reset_index()
    summary["band"] = [f"B{i}" for i in range(len(summary), 0, -1)]

    return summary[[
        'band', 'proba_segment', 'def_rate%', 'approval_rate%',
        'client', 'bad', 'good',
        '%bad vs total bad', '%bad cumsum',
        '%good vs total good',  '%good cumsum',
        'FPR', 'TPR', 'bad_scaled', 'good_scaled',
        'cumsum_client', 'cumsum_bad', 'cumsum_good',
    ]]
