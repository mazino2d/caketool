from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from caketool.utils.lib_utils import require_dependencies

if TYPE_CHECKING:
    import bigframes.pandas as bpd  # pyright: ignore[reportMissingImports]
    import pandas as pd
    import polars as pl  # pyright: ignore[reportMissingImports]
    from pyspark.sql import DataFrame as SparkDataFrame  # pyright: ignore[reportMissingImports]

    DataFrame = SparkDataFrame | bpd.DataFrame | pd.DataFrame | pl.DataFrame

# Backend type alias
Backend = Literal["spark", "bigframes", "pandas", "polars"]


@require_dependencies("pyspark")
def _generate_features_by_window_spark(
    df: SparkDataFrame,
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
    feature_prefix: str = "ft",
    key_col_default: str = "all",
) -> SparkDataFrame:
    """Spark implementation for generate_features_by_window."""
    import pyspark.sql.functions as F
    import pyspark.sql.types as T

    all_features = []
    if len(numeric_cols) + len(string_cols) + len(date_cols) == 0:
        raise ValueError("At least one of numeric_cols, string_cols or date_cols must be provided")

    # Pre-compute sets outside loop
    string_cat_cols = set(string_cols) | set(categorical_cols)

    temp_df = df
    for col in numeric_cols:
        temp_df = temp_df.withColumn(col, F.col(col).cast(T.DoubleType()))
    for col in boolean_cols:
        temp_df = temp_df.withColumn(col, F.col(col).cast(T.BooleanType()))

    if "__all__" in key_cols:
        temp_df = temp_df.withColumn("__all__", F.lit(key_col_default))

    if len(key_cols) > 1:
        temp_df = temp_df.cache()

    try:
        for key_col in key_cols:
            key_df = temp_df.withColumn(key_col, F.lower(F.col(key_col)))
            key_features = []
            for num_day in lookback_days:
                if num_day == 0:
                    window_df = key_df
                    lb_flag = "lifetime"
                elif num_day > 0:
                    window_df = (
                        key_df.withColumn("window_start", F.date_sub(F.col(fs_event_timestamp), num_day))
                        .withColumn("window_end", F.col(fs_event_timestamp))
                        .filter(
                            (F.col(report_date_col) >= F.col("window_start"))
                            & (F.col(report_date_col) < F.col("window_end"))
                        )
                        .orderBy(F.col(fs_event_timestamp))
                    )
                    lb_flag = f"d{num_day}"
                else:
                    raise ValueError("Lookback days must be a positive integer or zero (lifetime features)")

                agg_exprs = []
                for value_col in numeric_cols:
                    agg_exprs.extend(
                        [
                            F.min(value_col).alias(f"{value_col}_{lb_flag}_min"),
                            F.avg(value_col).alias(f"{value_col}_{lb_flag}_avg"),
                            F.expr(f"percentile_approx({value_col}, 0.25)").alias(f"{value_col}_{lb_flag}_p25"),
                            F.expr(f"percentile_approx({value_col}, 0.50)").alias(f"{value_col}_{lb_flag}_p50"),
                            F.expr(f"percentile_approx({value_col}, 0.75)").alias(f"{value_col}_{lb_flag}_p75"),
                            F.stddev(value_col).alias(f"{value_col}_{lb_flag}_std"),
                            F.max(value_col).alias(f"{value_col}_{lb_flag}_max"),
                            F.sum(value_col).alias(f"{value_col}_{lb_flag}_sum"),
                            F.count(value_col).alias(f"{value_col}_{lb_flag}_cnt"),
                            (F.max(value_col) - F.min(value_col)).alias(f"{value_col}_{lb_flag}_diff"),
                        ]
                    )

                for value_col in string_cat_cols:
                    agg_exprs.extend(
                        [
                            F.count(value_col).alias(f"{value_col}_{lb_flag}_cnt"),
                            F.countDistinct(value_col).alias(f"{value_col}_{lb_flag}_nunique"),
                            (F.countDistinct(value_col) / F.count(value_col)).alias(f"{value_col}_{lb_flag}_entropy"),
                        ]
                    )

                for date_col in date_cols:
                    agg_exprs.extend(
                        [
                            F.datediff(F.col(fs_event_timestamp), F.min(date_col)).alias(
                                f"{date_col}_{lb_flag}_firstdatediff"
                            ),
                            F.datediff(F.col(fs_event_timestamp), F.max(date_col)).alias(
                                f"{date_col}_{lb_flag}_lastdatediff"
                            ),
                            F.datediff(F.max(date_col), F.min(date_col)).alias(f"{date_col}_{lb_flag}_daysbetween"),
                        ]
                    )

                for bool_col in boolean_cols:
                    agg_exprs.extend(
                        [
                            F.sum(F.when(F.col(bool_col), 1).otherwise(0)).alias(f"{bool_col}_{lb_flag}_poscnt"),
                            F.avg(F.when(F.col(bool_col), 1).otherwise(0)).alias(f"{bool_col}_{lb_flag}_posratio"),
                        ]
                    )

                stats_df = window_df.groupBy(client_id_col, fs_event_timestamp, key_col).agg(*agg_exprs)

                pivot_df = stats_df.groupBy(client_id_col, fs_event_timestamp).pivot(key_col)
                pivot_exprs = []
                for value_col in set(stats_df.columns) - {client_id_col, fs_event_timestamp, key_col}:
                    pivot_exprs.append(F.first(value_col).alias(value_col))
                pivot_df = pivot_df.agg(*pivot_exprs)
                key_features.append(pivot_df)

            if key_features:
                key_result = key_features[0]
                for feat_df in key_features[1:]:
                    key_result = key_result.join(feat_df, on=[client_id_col, fs_event_timestamp], how="outer")
                all_features.append(key_result)

    finally:
        if len(key_cols) > 1:
            temp_df.unpersist()

    if not all_features:
        return df.select(client_id_col)

    result_df = all_features[0]
    for feature_df in all_features[1:]:
        result_df = result_df.join(feature_df, on=[client_id_col, fs_event_timestamp], how="outer")

    result_df = result_df.select(
        [
            F.col(c).alias(f"{feature_prefix}_{c}") if c not in [client_id_col, fs_event_timestamp] else c
            for c in result_df.columns
        ]
    )
    return result_df


@require_dependencies("bigframes")
def _generate_features_by_window_bq(
    df: bpd.DataFrame,
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
    feature_prefix: str = "ft",
    key_col_default: str = "all",
) -> bpd.DataFrame:
    """BigFrames (BigQuery) implementation for generate_features_by_window."""

    if len(numeric_cols) + len(string_cols) + len(date_cols) == 0:
        raise ValueError("At least one of numeric_cols, string_cols or date_cols must be provided")

    string_cat_cols = set(string_cols) | set(categorical_cols)
    all_features = []

    temp_df = df.copy()

    if "__all__" in key_cols:
        temp_df["__all__"] = key_col_default

    for key_col in key_cols:
        temp_df[key_col] = temp_df[key_col].str.lower()
        key_features = []

        for num_day in lookback_days:
            if num_day == 0:
                window_df = temp_df.copy()
                lb_flag = "lifetime"
            elif num_day > 0:
                window_df = temp_df.copy()
                window_df["_window_start"] = window_df[fs_event_timestamp] - bpd.to_timedelta(num_day, unit="D")
                window_df = window_df[
                    (window_df[report_date_col] >= window_df["_window_start"])
                    & (window_df[report_date_col] < window_df[fs_event_timestamp])
                ]
                lb_flag = f"d{num_day}"
            else:
                raise ValueError("Lookback days must be a positive integer or zero (lifetime features)")

            group_cols = [client_id_col, fs_event_timestamp, key_col]
            agg_dict = {}

            # Numeric aggregations
            for value_col in numeric_cols:
                agg_dict[f"{value_col}_{lb_flag}_min"] = (value_col, "min")
                agg_dict[f"{value_col}_{lb_flag}_max"] = (value_col, "max")
                agg_dict[f"{value_col}_{lb_flag}_avg"] = (value_col, "mean")
                agg_dict[f"{value_col}_{lb_flag}_sum"] = (value_col, "sum")
                agg_dict[f"{value_col}_{lb_flag}_std"] = (value_col, "std")
                agg_dict[f"{value_col}_{lb_flag}_cnt"] = (value_col, "count")

            # String/categorical count aggregations
            for value_col in string_cat_cols:
                agg_dict[f"{value_col}_{lb_flag}_cnt"] = (value_col, "count")
                agg_dict[f"{value_col}_{lb_flag}_nunique"] = (value_col, "nunique")

            # Boolean aggregations
            for bool_col in boolean_cols:
                agg_dict[f"{bool_col}_{lb_flag}_poscnt"] = (bool_col, "sum")
                agg_dict[f"{bool_col}_{lb_flag}_posratio"] = (bool_col, "mean")

            if agg_dict:
                stats_df = window_df.groupby(group_cols, as_index=False).agg(**agg_dict)

                # Calculate percentiles for numeric columns
                for value_col in numeric_cols:
                    for q, name in [(0.25, "p25"), (0.5, "p50"), (0.75, "p75")]:
                        quantile_df = window_df.groupby(group_cols)[value_col].quantile(q).reset_index()
                        quantile_df = quantile_df.rename(columns={value_col: f"{value_col}_{lb_flag}_{name}"})
                        stats_df = stats_df.merge(quantile_df, on=group_cols, how="left")

                # Calculate entropy for string/categorical columns
                for value_col in string_cat_cols:
                    cnt_col = f"{value_col}_{lb_flag}_cnt"
                    nunique_col = f"{value_col}_{lb_flag}_nunique"
                    entropy_col = f"{value_col}_{lb_flag}_entropy"
                    stats_df[entropy_col] = stats_df[nunique_col] / stats_df[cnt_col]

                # Calculate diff for numeric columns
                for value_col in numeric_cols:
                    min_col = f"{value_col}_{lb_flag}_min"
                    max_col = f"{value_col}_{lb_flag}_max"
                    diff_col = f"{value_col}_{lb_flag}_diff"
                    stats_df[diff_col] = stats_df[max_col] - stats_df[min_col]

            # Date aggregations (separate groupby due to different logic)
            if date_cols:
                date_agg_dict = {}
                for date_col in date_cols:
                    date_agg_dict[f"{date_col}_min"] = (date_col, "min")
                    date_agg_dict[f"{date_col}_max"] = (date_col, "max")

                date_stats = window_df.groupby(group_cols, as_index=False).agg(**date_agg_dict)

                for date_col in date_cols:
                    min_date = f"{date_col}_min"
                    max_date = f"{date_col}_max"
                    date_stats[f"{date_col}_{lb_flag}_firstdatediff"] = (
                        date_stats[fs_event_timestamp] - date_stats[min_date]
                    ).dt.days
                    date_stats[f"{date_col}_{lb_flag}_lastdatediff"] = (
                        date_stats[fs_event_timestamp] - date_stats[max_date]
                    ).dt.days
                    date_stats[f"{date_col}_{lb_flag}_daysbetween"] = (
                        date_stats[max_date] - date_stats[min_date]
                    ).dt.days
                    date_stats = date_stats.drop(columns=[min_date, max_date])

                if agg_dict:
                    stats_df = stats_df.merge(date_stats, on=group_cols, how="left")
                else:
                    stats_df = date_stats

            # Pivot by key_col
            pivot_cols = [c for c in stats_df.columns if c not in group_cols]
            pivot_df = stats_df.pivot_table(
                index=[client_id_col, fs_event_timestamp],
                columns=key_col,
                values=pivot_cols,
                aggfunc="first",
            )

            # Flatten column names
            pivot_df.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()

            key_features.append(pivot_df)

        if key_features:
            key_result = key_features[0]
            for feat_df in key_features[1:]:
                key_result = key_result.merge(
                    feat_df,
                    on=[client_id_col, fs_event_timestamp],
                    how="outer",
                )
            all_features.append(key_result)

    if not all_features:
        return df[[client_id_col]]

    result_df = all_features[0]
    for feature_df in all_features[1:]:
        result_df = result_df.merge(feature_df, on=[client_id_col, fs_event_timestamp], how="outer")

    # Rename columns with prefix
    rename_dict = {
        c: f"{feature_prefix}_{c}" for c in result_df.columns if c not in [client_id_col, fs_event_timestamp]
    }
    result_df = result_df.rename(columns=rename_dict)

    return result_df


def _generate_features_by_window_pandas(
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
    feature_prefix: str = "ft",
    key_col_default: str = "all",
) -> pd.DataFrame:
    """Pandas implementation for generate_features_by_window."""
    import pandas as pd

    if len(numeric_cols) + len(string_cols) + len(date_cols) == 0:
        raise ValueError("At least one of numeric_cols, string_cols or date_cols must be provided")

    string_cat_cols = set(string_cols) | set(categorical_cols)
    all_features = []

    temp_df = df.copy()

    if "__all__" in key_cols:
        temp_df["__all__"] = key_col_default

    for key_col in key_cols:
        temp_df[key_col] = temp_df[key_col].str.lower()
        key_features = []

        for num_day in lookback_days:
            if num_day == 0:
                window_df = temp_df.copy()
                lb_flag = "lifetime"
            elif num_day > 0:
                window_df = temp_df.copy()
                window_df["_window_start"] = window_df[fs_event_timestamp] - pd.Timedelta(days=num_day)
                window_df = window_df[
                    (window_df[report_date_col] >= window_df["_window_start"])
                    & (window_df[report_date_col] < window_df[fs_event_timestamp])
                ]
                lb_flag = f"d{num_day}"
            else:
                raise ValueError("Lookback days must be a positive integer or zero (lifetime features)")

            group_cols = [client_id_col, fs_event_timestamp, key_col]
            agg_dict = {}

            # Numeric aggregations
            for value_col in numeric_cols:
                agg_dict[f"{value_col}_{lb_flag}_min"] = (value_col, "min")
                agg_dict[f"{value_col}_{lb_flag}_max"] = (value_col, "max")
                agg_dict[f"{value_col}_{lb_flag}_avg"] = (value_col, "mean")
                agg_dict[f"{value_col}_{lb_flag}_sum"] = (value_col, "sum")
                agg_dict[f"{value_col}_{lb_flag}_std"] = (value_col, "std")
                agg_dict[f"{value_col}_{lb_flag}_cnt"] = (value_col, "count")

            # String/categorical count aggregations
            for value_col in string_cat_cols:
                agg_dict[f"{value_col}_{lb_flag}_cnt"] = (value_col, "count")
                agg_dict[f"{value_col}_{lb_flag}_nunique"] = (value_col, "nunique")

            # Boolean aggregations
            for bool_col in boolean_cols:
                agg_dict[f"{bool_col}_{lb_flag}_poscnt"] = (bool_col, "sum")
                agg_dict[f"{bool_col}_{lb_flag}_posratio"] = (bool_col, "mean")

            if agg_dict:
                stats_df = window_df.groupby(group_cols, as_index=False).agg(**agg_dict)

                # Calculate percentiles for numeric columns
                for value_col in numeric_cols:
                    for q, name in [(0.25, "p25"), (0.5, "p50"), (0.75, "p75")]:
                        quantile_df = window_df.groupby(group_cols)[value_col].quantile(q).reset_index()
                        quantile_df = quantile_df.rename(columns={value_col: f"{value_col}_{lb_flag}_{name}"})
                        stats_df = stats_df.merge(quantile_df, on=group_cols, how="left")

                # Calculate entropy for string/categorical columns
                for value_col in string_cat_cols:
                    cnt_col = f"{value_col}_{lb_flag}_cnt"
                    nunique_col = f"{value_col}_{lb_flag}_nunique"
                    entropy_col = f"{value_col}_{lb_flag}_entropy"
                    stats_df[entropy_col] = stats_df[nunique_col] / stats_df[cnt_col]

                # Calculate diff for numeric columns
                for value_col in numeric_cols:
                    min_col = f"{value_col}_{lb_flag}_min"
                    max_col = f"{value_col}_{lb_flag}_max"
                    diff_col = f"{value_col}_{lb_flag}_diff"
                    stats_df[diff_col] = stats_df[max_col] - stats_df[min_col]

            # Date aggregations
            if date_cols:
                date_agg_dict = {}
                for date_col in date_cols:
                    date_agg_dict[f"{date_col}_min"] = (date_col, "min")
                    date_agg_dict[f"{date_col}_max"] = (date_col, "max")

                date_stats = window_df.groupby(group_cols, as_index=False).agg(**date_agg_dict)

                for date_col in date_cols:
                    min_date = f"{date_col}_min"
                    max_date = f"{date_col}_max"
                    date_stats[f"{date_col}_{lb_flag}_firstdatediff"] = (
                        date_stats[fs_event_timestamp] - date_stats[min_date]
                    ).dt.days
                    date_stats[f"{date_col}_{lb_flag}_lastdatediff"] = (
                        date_stats[fs_event_timestamp] - date_stats[max_date]
                    ).dt.days
                    date_stats[f"{date_col}_{lb_flag}_daysbetween"] = (
                        date_stats[max_date] - date_stats[min_date]
                    ).dt.days
                    date_stats = date_stats.drop(columns=[min_date, max_date])

                if agg_dict:
                    stats_df = stats_df.merge(date_stats, on=group_cols, how="left")
                else:
                    stats_df = date_stats

            # Pivot by key_col
            pivot_cols = [c for c in stats_df.columns if c not in group_cols]
            pivot_df = stats_df.pivot_table(
                index=[client_id_col, fs_event_timestamp],
                columns=key_col,
                values=pivot_cols,
                aggfunc="first",
            )

            # Flatten column names
            pivot_df.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in pivot_df.columns]
            pivot_df = pivot_df.reset_index()

            key_features.append(pivot_df)

        if key_features:
            key_result = key_features[0]
            for feat_df in key_features[1:]:
                key_result = key_result.merge(
                    feat_df,
                    on=[client_id_col, fs_event_timestamp],
                    how="outer",
                )
            all_features.append(key_result)

    if not all_features:
        return df[[client_id_col]]

    result_df = all_features[0]
    for feature_df in all_features[1:]:
        result_df = result_df.merge(feature_df, on=[client_id_col, fs_event_timestamp], how="outer")

    # Rename columns with prefix
    rename_dict = {
        c: f"{feature_prefix}_{c}" for c in result_df.columns if c not in [client_id_col, fs_event_timestamp]
    }
    result_df = result_df.rename(columns=rename_dict)

    return result_df


@require_dependencies("polars")
def _generate_features_by_window_polars(
    df: pl.DataFrame,
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
    feature_prefix: str = "ft",
    key_col_default: str = "all",
) -> pl.DataFrame:
    """
    Generate aggregated features over time windows using Polars.

    Parameters
    ----------
    df : pl.DataFrame
        Input Polars DataFrame.
    client_id_col : str, optional
        Column name for client identifier (default: "user_id").
    report_date_col : str, optional
        Column name for report date (default: "report_date").
    fs_event_timestamp : str, optional
        Column name for feature store event timestamp (default: "fs_event_timestamp").
    key_cols : tuple[str, ...], optional
        Columns to group by and pivot. Use "__all__" for global aggregation (default: ("__all__",)).
    lookback_days : tuple[int, ...], optional
        Lookback windows in days. 0 = lifetime features (default: (0,)).
    numeric_cols : tuple[str, ...], optional
        Numeric columns for statistical aggregations (min, max, avg, sum, std).
    string_cols : tuple[str, ...], optional
        String columns for count/distinct aggregations.
    categorical_cols : tuple[str, ...], optional
        Categorical columns for count/distinct aggregations.
    date_cols : tuple[str, ...], optional
        Date columns for date difference aggregations.
    boolean_cols : tuple[str, ...], optional
        Boolean columns for positive count/ratio aggregations.
    feature_prefix : str, optional
        Prefix for generated feature names (default: "ft").
    key_col_default : str, optional
        Default value for "__all__" key column (default: "all").

    Returns
    -------
    pl.DataFrame
        DataFrame with generated features.

    Warning
    -------
    Approximate aggregations (results may vary slightly):
    - Percentiles (p25, p50, p75) use `quantile` method

    Examples
    --------
    >>> import polars as pl
    >>> features_df = generate_features_by_window_polars(
    ...     df,
    ...     client_id_col="user_id",
    ...     lookback_days=(0, 7, 30),
    ...     numeric_cols=("amount", "quantity"),
    ... )
    """
    from datetime import timedelta

    if len(numeric_cols) + len(string_cols) + len(date_cols) == 0:
        raise ValueError("At least one of numeric_cols, string_cols or date_cols must be provided")

    string_cat_cols = set(string_cols) | set(categorical_cols)
    all_features = []

    temp_df = df.clone()

    if "__all__" in key_cols:
        temp_df = temp_df.with_columns(pl.lit(key_col_default).alias("__all__"))

    for key_col in key_cols:
        temp_df = temp_df.with_columns(pl.col(key_col).str.to_lowercase())
        key_features = []

        for num_day in lookback_days:
            if num_day == 0:
                window_df = temp_df.clone()
                lb_flag = "lifetime"
            elif num_day > 0:
                window_df = temp_df.with_columns(
                    (pl.col(fs_event_timestamp) - timedelta(days=num_day)).alias("_window_start")
                ).filter(
                    (pl.col(report_date_col) >= pl.col("_window_start"))
                    & (pl.col(report_date_col) < pl.col(fs_event_timestamp))
                )
                lb_flag = f"d{num_day}"
            else:
                raise ValueError("Lookback days must be a positive integer or zero (lifetime features)")

            group_cols = [client_id_col, fs_event_timestamp, key_col]
            agg_exprs = []

            # Numeric aggregations
            for value_col in numeric_cols:
                agg_exprs.extend(
                    [
                        pl.col(value_col).min().alias(f"{value_col}_{lb_flag}_min"),
                        pl.col(value_col).max().alias(f"{value_col}_{lb_flag}_max"),
                        pl.col(value_col).mean().alias(f"{value_col}_{lb_flag}_avg"),
                        pl.col(value_col).sum().alias(f"{value_col}_{lb_flag}_sum"),
                        pl.col(value_col).std().alias(f"{value_col}_{lb_flag}_std"),
                        pl.col(value_col).count().alias(f"{value_col}_{lb_flag}_cnt"),
                        (pl.col(value_col).max() - pl.col(value_col).min()).alias(f"{value_col}_{lb_flag}_diff"),
                        pl.col(value_col).quantile(0.25).alias(f"{value_col}_{lb_flag}_p25"),
                        pl.col(value_col).quantile(0.50).alias(f"{value_col}_{lb_flag}_p50"),
                        pl.col(value_col).quantile(0.75).alias(f"{value_col}_{lb_flag}_p75"),
                    ]
                )

            # String/categorical count aggregations
            for value_col in string_cat_cols:
                agg_exprs.extend(
                    [
                        pl.col(value_col).count().alias(f"{value_col}_{lb_flag}_cnt"),
                        pl.col(value_col).n_unique().alias(f"{value_col}_{lb_flag}_nunique"),
                        (pl.col(value_col).n_unique() / pl.col(value_col).count()).alias(
                            f"{value_col}_{lb_flag}_entropy"
                        ),
                    ]
                )

            # Boolean aggregations
            for bool_col in boolean_cols:
                agg_exprs.extend(
                    [
                        pl.col(bool_col).sum().alias(f"{bool_col}_{lb_flag}_poscnt"),
                        pl.col(bool_col).mean().alias(f"{bool_col}_{lb_flag}_posratio"),
                    ]
                )

            # Date aggregations
            for date_col in date_cols:
                agg_exprs.extend(
                    [
                        (pl.col(fs_event_timestamp) - pl.col(date_col).min())
                        .dt.total_days()
                        .alias(f"{date_col}_{lb_flag}_firstdatediff"),
                        (pl.col(fs_event_timestamp) - pl.col(date_col).max())
                        .dt.total_days()
                        .alias(f"{date_col}_{lb_flag}_lastdatediff"),
                        (pl.col(date_col).max() - pl.col(date_col).min())
                        .dt.total_days()
                        .alias(f"{date_col}_{lb_flag}_daysbetween"),
                    ]
                )

            if agg_exprs:
                stats_df = window_df.group_by(group_cols).agg(agg_exprs)

                # Pivot by key_col
                value_cols = [c for c in stats_df.columns if c not in group_cols]
                pivot_df = stats_df.pivot(
                    on=key_col,
                    index=[client_id_col, fs_event_timestamp],
                    values=value_cols,
                    aggregate_function="first",
                )

                key_features.append(pivot_df)

        if key_features:
            key_result = key_features[0]
            for feat_df in key_features[1:]:
                key_result = key_result.join(feat_df, on=[client_id_col, fs_event_timestamp], how="outer")
            all_features.append(key_result)

    if not all_features:
        return df.select(client_id_col)

    result_df = all_features[0]
    for feature_df in all_features[1:]:
        result_df = result_df.join(feature_df, on=[client_id_col, fs_event_timestamp], how="outer")

    # Rename columns with prefix
    rename_dict = {
        c: f"{feature_prefix}_{c}" for c in result_df.columns if c not in [client_id_col, fs_event_timestamp]
    }
    result_df = result_df.rename(rename_dict)

    return result_df


# Backend function registry
_BACKEND_FUNCTIONS = {
    "spark": _generate_features_by_window_spark,
    "bigframes": _generate_features_by_window_bq,
    "pandas": _generate_features_by_window_pandas,
    "polars": _generate_features_by_window_polars,
}


def generate_features_by_window(
    df: DataFrame,
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
    feature_prefix: str = "ft",
    key_col_default: str = "all",
    backend: Backend = "pandas",
) -> DataFrame:
    """
    Generate aggregated features over time windows grouped by key columns.

    Supports multiple backends: pandas, polars, spark, and bigframes (BigQuery).
    Generates statistical aggregations for numeric, string, categorical, date,
    and boolean columns over specified lookback windows.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame (pandas, polars, Spark, or BigFrames).
    client_id_col : str, optional
        Column name for client identifier (default: "user_id").
    report_date_col : str, optional
        Column name for report date (default: "report_date").
    fs_event_timestamp : str, optional
        Column name for feature store event timestamp (default: "fs_event_timestamp").
    key_cols : tuple[str, ...], optional
        Columns to group by and pivot. Use "__all__" for global aggregation
        without grouping by a specific column (default: ("__all__",)).
    lookback_days : tuple[int, ...], optional
        Lookback windows in days. Use 0 for lifetime features that aggregate
        all historical data (default: (0,)).
    numeric_cols : tuple[str, ...], optional
        Numeric columns for statistical aggregations. Generates:
        min, max, avg, sum, std, cnt, diff, p25, p50, p75.
    string_cols : tuple[str, ...], optional
        String columns for count aggregations. Generates: cnt, nunique, entropy.
    categorical_cols : tuple[str, ...], optional
        Categorical columns for count aggregations. Generates: cnt, nunique, entropy.
    date_cols : tuple[str, ...], optional
        Date columns for date difference aggregations. Generates:
        firstdatediff, lastdatediff, daysbetween.
    boolean_cols : tuple[str, ...], optional
        Boolean columns for positive count/ratio aggregations. Generates:
        poscnt, posratio.
    feature_prefix : str, optional
        Prefix for generated feature names (default: "ft").
    key_col_default : str, optional
        Default value for "__all__" key column (default: "all").
    backend : {"pandas", "polars", "spark", "bigframes"}, optional
        Backend to use for feature generation (default: "pandas").

    Returns
    -------
    DataFrame
        DataFrame with generated features, indexed by client_id_col and
        fs_event_timestamp. Feature columns are prefixed with feature_prefix.

    Raises
    ------
    ValueError
        If no feature columns are provided or if backend is not supported.

    Warning
    -------
    Approximate aggregations (results may vary slightly):
    - Percentiles (p25, p50, p75) use approximate methods
      (percentile_approx for Spark, quantile for others)

    Note
    ----
    Backend requirements:
    - pandas: No additional dependencies
    - polars: pip install polars
    - spark: pip install pyspark
    - bigframes: pip install bigframes

    Examples
    --------
    >>> import pandas as pd
    >>> features_df = generate_features_by_window(
    ...     df,
    ...     client_id_col="user_id",
    ...     lookback_days=(0, 7, 30),
    ...     numeric_cols=("amount", "quantity"),
    ...     categorical_cols=("category",),
    ...     backend="pandas",
    ... )

    >>> # Using Polars backend
    >>> import polars as pl
    >>> features_df = generate_features_by_window(
    ...     df,
    ...     lookback_days=(0, 7),
    ...     numeric_cols=("amount",),
    ...     backend="polars",
    ... )

    >>> # Using Spark backend
    >>> features_df = generate_features_by_window(
    ...     spark_df,
    ...     lookback_days=(0, 30, 90),
    ...     numeric_cols=("transaction_amount",),
    ...     string_cols=("merchant_name",),
    ...     backend="spark",
    ... )

    >>> # Using BigFrames backend for BigQuery
    >>> import bigframes.pandas as bpd
    >>> bpd.options.bigquery.project = "my-project"
    >>> df = bpd.read_gbq("SELECT * FROM my_table")
    >>> features_df = generate_features_by_window(
    ...     df,
    ...     numeric_cols=("amount",),
    ...     backend="bigframes",
    ... )
    """
    if backend not in _BACKEND_FUNCTIONS:
        supported = ", ".join(_BACKEND_FUNCTIONS.keys())
        raise ValueError(f"Unsupported backend: {backend}. Supported: {supported}")

    return _BACKEND_FUNCTIONS[backend](
        df=df,
        client_id_col=client_id_col,
        report_date_col=report_date_col,
        fs_event_timestamp=fs_event_timestamp,
        key_cols=key_cols,
        lookback_days=lookback_days,
        numeric_cols=numeric_cols,
        string_cols=string_cols,
        categorical_cols=categorical_cols,
        date_cols=date_cols,
        boolean_cols=boolean_cols,
        feature_prefix=feature_prefix,
        key_col_default=key_col_default,
    )
