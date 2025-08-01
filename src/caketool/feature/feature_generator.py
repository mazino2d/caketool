from typing import List
import unicodedata
import warnings
import pandas as pd
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql.window import Window

warnings.filterwarnings("ignore")
pd.DataFrame.iteritems = pd.DataFrame.items


def generate_features_by_window(
    df: SparkDataFrame,
    client_id_col: str = "cake_user_id",
    report_date_col: str = "report_date",
    fs_event_timestamp: str = "fs_event_timestamp",
    key_cols: List[str] = ["__all__"],
    lookback_days: List[int] = [0],
    numeric_cols: List[str] = [],
    string_cols: List[str] = [],
    categorical_cols: List[str] = [],
    list_cols: List[str] = [],
    date_cols: List[str] = [],
    boolean_cols: List[str] = [],
    feature_prefix: str = "ft",
    key_col_default: str = "all",
):
    all_features = []
    if len(numeric_cols) + len(string_cols) + len(date_cols) == 0:
        raise ValueError("At least one of numeric_cols, categorical_cols or date_cols must be provided")

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
                        key_df
                        .withColumn("window_start", F.date_sub(F.col(fs_event_timestamp), num_day))
                        .withColumn("window_end", F.col(fs_event_timestamp))
                        .filter(
                            (F.col(report_date_col) >= F.col("window_start")) &
                            (F.col(report_date_col) < F.col("window_end"))
                        )
                        .orderBy(F.col(fs_event_timestamp))
                    )
                    lb_flag = f"d{num_day}"
                else:
                    raise ValueError("Lookback days must be a positive integer or zero (lifetime features)")

                agg_exprs = []
                for value_col in numeric_cols:
                    agg_exprs.extend([
                        F.min(value_col).alias(f"{value_col}_{lb_flag}_min"),
                        F.avg(value_col).alias(f"{value_col}_{lb_flag}_avg"),
                        F.expr(f"percentile_approx({value_col}, 0.25)").alias(f"{value_col}_{lb_flag}_p25"),
                        F.expr(f"percentile_approx({value_col}, 0.50)").alias(f"{value_col}_{lb_flag}_p50"),
                        F.expr(f"percentile_approx({value_col}, 0.75)").alias(f"{value_col}_{lb_flag}_p75"),
                        F.stddev(value_col).alias(f"{value_col}_{lb_flag}_std"),
                        F.max(value_col).alias(f"{value_col}_{lb_flag}_max"),
                        F.sum(value_col).alias(f"{value_col}_{lb_flag}_sum"),
                        F.count(value_col).alias(f"{value_col}_{lb_flag}_cnt"),
                        F.skewness(value_col).alias(f"{value_col}_{lb_flag}_skew"),
                        F.kurtosis(value_col).alias(f"{value_col}_{lb_flag}_kurt"),
                        (F.max(value_col) - F.min(value_col)).alias(f"{value_col}_{lb_flag}_diff"),
                    ])

                for value_col in set(string_cols + categorical_cols):
                    agg_exprs.extend([
                        F.count(value_col).alias(f"{value_col}_{lb_flag}_cnt"),
                        F.countDistinct(value_col).alias(f"{value_col}_{lb_flag}_nunique"),
                        (F.countDistinct(value_col) / F.count(value_col)).alias(f"{value_col}_{lb_flag}_entropy"),
                    ])

                for value_col in list_cols:
                    agg_exprs.extend([
                        F.size(F.collect_set(value_col)).alias(f"{value_col}_{lb_flag}_nunique"),
                    ])

                for date_col in date_cols:
                    agg_exprs.extend([
                        F.datediff(F.col(fs_event_timestamp), F.min(date_col)).alias(f"{date_col}_{lb_flag}_firstdatediff"),
                        F.datediff(F.col(fs_event_timestamp), F.max(date_col)).alias(f"{date_col}_{lb_flag}_lastdatediff"),
                        F.datediff(F.max(date_col), F.min(date_col)).alias(f"{date_col}_{lb_flag}_daysbetween"),
                    ])

                for bool_col in boolean_cols:
                    agg_exprs.extend([
                        F.sum(F.when(F.col(bool_col), 1).otherwise(0)).alias(f"{bool_col}_{lb_flag}_poscnt"),
                        F.avg(F.when(F.col(bool_col), 1).otherwise(0)).alias(f"{bool_col}_{lb_flag}_posratio"),
                    ])

                stats_df = window_df.groupBy(client_id_col, fs_event_timestamp, key_col).agg(*agg_exprs)

                # SPARK LEGACY MODE -- SPARK < 3.5.0
                for value_col in categorical_cols:
                    count_df = (
                        window_df
                        .groupBy(client_id_col, fs_event_timestamp, key_col, value_col)
                        .agg(F.count("*").alias("cnt"))
                    )

                    window_spec = Window.partitionBy(client_id_col, fs_event_timestamp, key_col) \
                                        .orderBy(F.desc("cnt"), F.asc(value_col))

                    mode_df = (
                        count_df
                        .withColumn("rn", F.row_number().over(window_spec))
                        .filter(F.col("rn") == 1)
                        .select(
                            client_id_col,
                            fs_event_timestamp,
                            key_col,
                            F.col(value_col).alias(f"{value_col}_d{num_day}_mode"),
                            F.col("cnt").alias(f"{value_col}_d{num_day}_mode_cnt")
                        )
                    )

                    stats_df = stats_df.join(mode_df, on=[client_id_col, fs_event_timestamp, key_col], how="left")

                # for value_col in categorical_cols:
                #     agg_exprs.extend([
                #         F.mode(value_col, ignorenulls=True).alias(f"{value_col}_{lb_flag}_mode"),
                #     ])
                # SPARK LEGACY MODE -- SPARK < 3.5.0

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
        [F.col(c).alias(f"{feature_prefix}_{c}") if c not in [client_id_col, fs_event_timestamp] else c for c in result_df.columns]
    )
    return result_df


def standardize_string_columns(df: SparkDataFrame) -> SparkDataFrame:
    def remove_accents(input_str: str):
        if input_str is None:
            return None
        nfkd_form = unicodedata.normalize('NFKD', input_str)
        acccent_removed_form = "".join(
            [c for c in nfkd_form if not unicodedata.combining(c)])
        return acccent_removed_form.lower().replace('đ', 'd')

    remove_accents_udf = F.udf(remove_accents, F.StringType())
    string_cols = [f.name for f in df.schema.fields if isinstance(
        f.dataType, F.StringType)]
    print(string_cols)
    for c in string_cols:
        df = df.withColumn(c, remove_accents_udf(F.lower(F.trim(F.col(c)))))
    return df
