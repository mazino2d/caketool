from collections.abc import Mapping, Sequence

import pandas as pd

# Pivot label used for rows whose key column is null
NULL_KEY = "null"

# Aggregations available per column type, in default output order
NUMERIC_AGGS = ("min", "max", "avg", "sum", "std", "cnt", "diff", "p25", "p50", "p75")
STRING_AGGS = ("cnt", "nunique", "unique_ratio")
BOOLEAN_AGGS = ("poscnt", "posratio")
DATE_AGGS = ("firstdatediff", "lastdatediff", "daysbetween")

_AGGS_BY_TYPE = {"numeric": NUMERIC_AGGS, "string": STRING_AGGS, "boolean": BOOLEAN_AGGS, "date": DATE_AGGS}
_QUANTILES = {"p25": 0.25, "p50": 0.5, "p75": 0.75}

# pandas groupby function behind each aggregation, by column type. Derived
# aggregations list the base aggregations they are computed from.
_BASE_FUNCS = {
    "numeric": {"min": "min", "max": "max", "avg": "mean", "sum": "sum", "std": "std", "cnt": "count"},
    "string": {"cnt": "count", "nunique": "nunique"},
    "boolean": {"poscnt": "sum", "posratio": "mean"},
    "date": {"min": "min", "max": "max"},
}
_DERIVED_FROM = {
    "numeric": {"diff": ("min", "max")},
    "string": {"unique_ratio": ("cnt", "nunique")},
    "date": {"firstdatediff": ("min",), "lastdatediff": ("max",), "daysbetween": ("min", "max")},
}


def _resolve_aggs(
    numeric_cols: Sequence[str],
    string_cols: Sequence[str],
    categorical_cols: Sequence[str],
    date_cols: Sequence[str],
    boolean_cols: Sequence[str],
    aggs: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return ``{column: (column_type, aggregations)}`` in output order."""
    col_types: dict[str, str] = {}
    for col_type, cols in (
        ("numeric", numeric_cols),
        ("string", (*string_cols, *categorical_cols)),
        ("boolean", boolean_cols),
        ("date", date_cols),
    ):
        for col in cols:
            col_types.setdefault(col, col_type)

    if not col_types:
        raise ValueError(
            "At least one of numeric_cols, string_cols, categorical_cols, date_cols or boolean_cols must be provided"
        )

    aggs = dict(aggs or {})
    undeclared = [col for col in aggs if col not in col_types]
    if undeclared:
        raise ValueError(
            f"Columns in aggs must also be listed in numeric_cols, string_cols, categorical_cols, date_cols "
            f"or boolean_cols: {undeclared}"
        )

    plan: dict[str, tuple[str, tuple[str, ...]]] = {}
    for col, col_type in col_types.items():
        allowed = _AGGS_BY_TYPE[col_type]
        requested = aggs.get(col, allowed)
        if isinstance(requested, str):
            requested = (requested,)
        invalid = [agg for agg in requested if agg not in allowed]
        if invalid:
            raise ValueError(f"Invalid aggregations for {col_type} column {col!r}: {invalid}. Allowed: {list(allowed)}")
        requested = tuple(dict.fromkeys(requested))
        if requested:
            plan[col] = (col_type, requested)

    if not plan:
        raise ValueError("aggs disables every aggregation; nothing to compute")
    return plan


def _aggregate_window(
    window_df: pd.DataFrame,
    group_cols: list[str],
    fs_event_timestamp: str,
    plan: dict[str, tuple[str, tuple[str, ...]]],
    lb_flag: str,
) -> pd.DataFrame:
    """Aggregate one lookback window. Returns one row per group, indexed by ``group_cols``."""
    agg_spec: dict[str, tuple[str, str]] = {}
    quantile_cols: dict[str, list[str]] = {}

    def base_name(col: str, base: str) -> str:
        return f"{col}_{lb_flag}_{base}"

    for col, (col_type, requested) in plan.items():
        needed: list[str] = []
        for agg in requested:
            if agg in _QUANTILES:
                quantile_cols.setdefault(col, []).append(agg)
            else:
                needed.extend(_DERIVED_FROM.get(col_type, {}).get(agg, (agg,)))
        for base in dict.fromkeys(needed):
            agg_spec[base_name(col, base)] = (col, _BASE_FUNCS[col_type][base])

    grouped = window_df.groupby(group_cols)
    stats = grouped.agg(**agg_spec) if agg_spec else grouped.size().to_frame("__size__")

    if quantile_cols:
        qs = sorted({_QUANTILES[name] for names in quantile_cols.values() for name in names})
        quantiles = grouped[list(quantile_cols)].quantile(qs).unstack(level=-1)
        for col, names in quantile_cols.items():
            for name in names:
                stats[base_name(col, name)] = quantiles[(col, _QUANTILES[name])]

    snapshot = stats.index.get_level_values(fs_event_timestamp)
    output: dict[str, pd.Series] = {}
    for col, (col_type, requested) in plan.items():
        for agg in requested:
            name = base_name(col, agg)
            if col_type == "numeric" and agg == "diff":
                output[name] = stats[base_name(col, "max")] - stats[base_name(col, "min")]
            elif col_type == "string" and agg == "unique_ratio":
                output[name] = stats[base_name(col, "nunique")] / stats[base_name(col, "cnt")]
            elif col_type == "date":
                first, last = stats.get(base_name(col, "min")), stats.get(base_name(col, "max"))
                if agg == "firstdatediff":
                    output[name] = (snapshot - first).dt.days
                elif agg == "lastdatediff":
                    output[name] = (snapshot - last).dt.days
                else:
                    output[name] = (last - first).dt.days
            else:
                output[name] = stats[name]

    return pd.DataFrame(output, index=stats.index)


def generate_features_by_window(
    df: pd.DataFrame,
    client_id_col: str = "user_id",
    report_date_col: str = "report_date",
    fs_event_timestamp: str = "fs_event_timestamp",
    key_cols: tuple[str, ...] = ("__all__",),
    lookback_days: tuple[int, ...] = (0,),
    numeric_cols: tuple[str, ...] = (),
    string_cols: tuple[str, ...] = (),
    categorical_cols: tuple[str, ...] = (),
    date_cols: tuple[str, ...] = (),
    boolean_cols: tuple[str, ...] = (),
    aggs: Mapping[str, Sequence[str]] | None = None,
    feature_prefix: str = "ft",
    key_col_default: str = "all",
) -> pd.DataFrame:
    """
    Generate aggregated features over time windows grouped by key columns.

    Generates statistical aggregations for numeric, string, categorical, date
    and boolean columns over the given lookback windows. By default every
    aggregation of a column's type is computed; pass ``aggs`` to compute only
    the ones you need, which is much faster on wide inputs.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level input, one row per event.
    client_id_col : str, optional
        Column name for client identifier (default: "user_id").
    report_date_col : str, optional
        Column name for the event date (default: "report_date").
    fs_event_timestamp : str, optional
        Column name for the snapshot (feature store event) timestamp
        (default: "fs_event_timestamp").
    key_cols : tuple[str, ...], optional
        Columns to group by and pivot. Use "__all__" for global aggregation
        without grouping by a specific column (default: ("__all__",)).
        Key values are cast to lowercase strings; null keys are pivoted under
        the "null" label.
    lookback_days : tuple[int, ...], optional
        Lookback windows in days. Use 0 for lifetime features that aggregate
        all history strictly before ``fs_event_timestamp`` (default: (0,)).
    numeric_cols : tuple[str, ...], optional
        Numeric columns. Aggregations: min, max, avg, sum, std, cnt, diff
        (max - min), p25, p50, p75.
    string_cols : tuple[str, ...], optional
        String columns. Aggregations: cnt, nunique, unique_ratio
        (nunique / cnt). Nulls are not counted.
    categorical_cols : tuple[str, ...], optional
        Categorical columns; same aggregations as ``string_cols``.
    date_cols : tuple[str, ...], optional
        Date columns. Aggregations: firstdatediff, lastdatediff (days from the
        earliest / latest date to the snapshot), daysbetween (days between them).
    boolean_cols : tuple[str, ...], optional
        Boolean columns. Aggregations: poscnt, posratio. Nulls are skipped
        (posratio = True / non-null count).
    aggs : Mapping[str, Sequence[str]], optional
        Aggregations to compute per column, e.g.
        ``{"amount": ["sum", "max"], "merchant": ["nunique"]}``. Each column
        must also be listed in one of the ``*_cols`` arguments, which sets its
        type. Columns not in ``aggs`` get every aggregation of their type; an
        empty list skips the column. Default ``None`` (all aggregations).
    feature_prefix : str, optional
        Prefix for generated feature names (default: "ft").
    key_col_default : str, optional
        Default value for "__all__" key column (default: "all").

    Returns
    -------
    pd.DataFrame
        One row per (client_id_col, fs_event_timestamp) with feature columns
        named ``{feature_prefix}_{key}_{column}_{window}_{aggregation}``,
        where window is ``lifetime`` or ``d{days}``.

    Raises
    ------
    ValueError
        If no feature columns are given, ``aggs`` names an undeclared column
        or an aggregation that does not fit the column type, or a lookback
        is negative.

    Examples
    --------
    >>> features_df = generate_features_by_window(
    ...     df,
    ...     client_id_col="user_id",
    ...     lookback_days=(0, 7, 30),
    ...     numeric_cols=("amount", "quantity"),
    ...     categorical_cols=("category",),
    ... )

    >>> # Only the aggregations you need
    >>> features_df = generate_features_by_window(
    ...     df,
    ...     lookback_days=(7, 30),
    ...     numeric_cols=("amount", "quantity"),
    ...     categorical_cols=("category",),
    ...     aggs={"amount": ["sum", "max"], "quantity": ["avg"], "category": ["nunique"]},
    ... )
    """
    plan = _resolve_aggs(numeric_cols, string_cols, categorical_cols, date_cols, boolean_cols, aggs)
    if any(num_day < 0 for num_day in lookback_days):
        raise ValueError("Lookback days must be a positive integer or zero (lifetime features)")

    used_cols = [client_id_col, report_date_col, fs_event_timestamp, *plan]
    used_cols += [col for col in key_cols if col != "__all__"]
    temp_df = df[list(dict.fromkeys(used_cols))].copy()
    # Nullable boolean so sum/mean skip missing values
    for col, (col_type, _) in plan.items():
        if col_type == "boolean":
            temp_df[col] = temp_df[col].astype("boolean")

    if "__all__" in key_cols:
        temp_df["__all__"] = key_col_default

    history = temp_df[temp_df[report_date_col] < temp_df[fs_event_timestamp]]
    id_cols = [client_id_col, fs_event_timestamp]
    all_features = []

    for key_col in key_cols:
        key_df = history.assign(**{key_col: history[key_col].astype("string").str.lower().fillna(NULL_KEY)})
        group_cols = [*id_cols, key_col]

        for num_day in lookback_days:
            if num_day == 0:
                window_df, lb_flag = key_df, "lifetime"
            else:
                window_start = key_df[fs_event_timestamp] - pd.Timedelta(days=num_day)
                window_df, lb_flag = key_df[key_df[report_date_col] >= window_start], f"d{num_day}"

            stats_df = _aggregate_window(window_df, group_cols, fs_event_timestamp, plan, lb_flag)
            # unstack (unlike pivot_table) keeps all-NaN feature columns, so the
            # output schema does not depend on the data.
            pivot_df = stats_df.unstack(key_col)
            pivot_df.columns = [f"{key_value}_{feature}" for feature, key_value in pivot_df.columns]
            all_features.append(pivot_df)

    if not all_features:
        return df[[client_id_col]]

    result_df = pd.concat(all_features, axis=1, join="outer").reset_index()
    rename_dict = {c: f"{feature_prefix}_{c}" for c in result_df.columns if c not in id_cols}
    return result_df.rename(columns=rename_dict)
