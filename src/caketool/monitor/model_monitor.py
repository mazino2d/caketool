from typing import List, Set, Dict, Union
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from caketool.utils import str_utils, num_utils


class ModelMonitor:

    def __init__(self, project, location, dataset="model_motinor") -> None:
        self.project = project
        self.location = location
        self.dataset = dataset
        self.MISSING = 'cake.miss'
        self.OTHER = 'cake.other'
        self.bq_client = bigquery.Client(project=self.project, location=self.location)
        self.ID_COLS = ["score_type", "dataset_type", "version_type", "version"]
        self.ID_SCHEMA = [
            bigquery.SchemaField(name, 'STRING', 'REQUIRED')
            for name in self.ID_COLS
        ]

    def normalize_data(
        self, df: pd.DataFrame, inplace: bool = False,
        cate_missing_values: Set[str] = {'-1', '-100', 'unknown', ''}
    ) -> pd.DataFrame:
        if not inplace:
            df = df.copy()
        # Fill missing value
        df = df.fillna(-100)
        # Norm categorical feature type
        categorical_features: List[str] = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, KeyError):
                categorical_features.append(col)
        df[categorical_features] = df[categorical_features].apply(lambda x: x.astype(str).str.lower())

        # Find numerical features
        numerical_features: List[str] = df.select_dtypes([int, float]).columns
        # Handle infinity value
        for col in list(set(df[numerical_features].columns.to_series()[np.isinf(df[numerical_features]).any()])):
            df[col] = df[col].apply(lambda x: -100 if x == np.inf else x)
        # Norm float feature
        float_features: List[str] = []
        for col in numerical_features:
            if df[col].fillna(0).nunique() > round(df[col].fillna(0)).nunique():
                float_features.append(col)
        df[float_features] = df[float_features].astype(float)
        # Norm int feature
        int_features = list(set(df.columns) - set(categorical_features) - set(float_features))
        df[int_features] = df[int_features].astype(int)
        # Handle numeric columns
        for col in numerical_features:
            df[col] = df[col].apply(lambda x: -100 if x < 0 else x)
        # Handle categorical columns
        for col in categorical_features:
            df[col] = df[col].apply(
                lambda x: self.MISSING
                if x in cate_missing_values
                else str_utils.remove_vn_diacritics(x).lower().strip()
            )
        df.__is_norm = True
        return df

    def create_bin_data(
        self,
        df: pd.DataFrame,
        n_bins=10
    ) -> Dict[str, List[Union[float, str]]]:
        if not self._check_norm(df):
            raise Exception("DataFrame has not been normalized yet. Please use self.normalize_data(df)")
        numerical_features = set(df.select_dtypes([int, float]).columns)
        categorical_features = list(df.select_dtypes([object]).columns)
        bin_thresholds = []
        # Bin num features
        for f in numerical_features:
            series = df[f]
            series = series[(series > 0) & (~series.isna())]
            if len(series) == 0:
                bins = np.array([])
            else:
                percentage = np.linspace(0, 100, n_bins + 1)
                bins = np.percentile(series, percentage)
                bins = [num_utils.round(e, series.dtype) for e in bins]
                bins = np.unique([0.0, *bins])
                if len(bins) >= 2:
                    bins[-1] = bins[-1] + 1e-10
            bin_thresholds.append([f, str(series.dtype).lower(), bins.tolist()])
        # Bin cate features
        for f in categorical_features:
            series = df[f]
            bins = series[series != self.MISSING].value_counts()[:n_bins].index.to_list()
            bins = sorted(set([self.MISSING, self.OTHER, *bins]))
            bin_thresholds.append([f, "string", bins])

        df_bins = pd.DataFrame(bin_thresholds, columns=["feature_name", "type", "bins"])
        return df_bins.sort_values("feature_name").reset_index(drop=True)

    def store_bin_data(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_bins: pd.DataFrame,
        bq_table_name="feature_bins"
    ):
        df_bins["bins"] = df_bins["bins"].apply(lambda ls: list(map(str, ls)))
        self._store_df(
            score_type, dataset_type, version_type, version, df_bins, bq_table_name,
            [
                bigquery.SchemaField("feature_name", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("type", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("bins", 'STRING', 'REPEATED'),
            ]
        )

    def load_bin_data(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        bq_table_name="feature_bins"
    ):
        table_id = f"{self.project}.{self.dataset}.{bq_table_name}"
        df_bins: pd.DataFrame = self.bq_client.query(f"""
            SELECT feature_name, type, bins FROM {table_id}
            WHERE score_type = '{score_type}'
            AND dataset_type = '{dataset_type}'
            AND version_type = '{version_type}'
            AND version = '{version}'
        """).to_dataframe()

        def cvt_bins(r):
            if r["type"] == "string":
                return r["bins"]
            else:
                return [float(e) for e in r["bins"]]
        df_bins["bins"] = df_bins.apply(cvt_bins, axis=1)
        return df_bins.sort_values("feature_name").reset_index(drop=True)

    def calc_feature_distribution(
        self,
        df: pd.DataFrame,
        df_bins: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        if not self._check_norm(df):
            raise Exception("DataFrame has not been normalized yet. Please use self.normalize_data(df)")
        bin_thresholds: Dict[str, np.ndarray] = dict(zip(df_bins.feature_name, df_bins.bins))
        hists = []
        categorical_features = [k for k, v in bin_thresholds.items() if len(v) > 0 and isinstance(v[0], str)]
        numerical_features = [k for k, v in bin_thresholds.items() if len(v) > 0 and not isinstance(v[0], str)]

        for f in numerical_features:
            series = df[f]
            hist, _ = np.histogram(series, [-np.inf, *bin_thresholds[f], np.inf])
            segments = [f"missing", *self._cvt_bins2labels(bin_thresholds[f]), f"other"]
            segments = [". ".join(e) for e in zip(str_utils.UPPER_ALPHABET, segments)]
            hists.append([f, segments, hist, hist.sum(), hist / hist.sum()])
        for f in categorical_features:
            series = df[f]
            bins = bin_thresholds[f]
            series = series.apply(lambda x: x if x in bins else self.OTHER)
            vc = series.value_counts()
            for bin_name in bins:
                if bin_name not in vc.index:
                    vc.loc[bin_name] = 0
            vc = vc.reindex(bins)
            segments = [". ".join(e) for e in zip(str_utils.UPPER_ALPHABET, bins)]
            hists.append([f, segments, vc, vc.sum(), vc / vc.sum()])

        return pd.DataFrame(hists, columns=["feature_name", "segment", "count", "total", "percent"]).explode(["segment", "count", "percent"])
    
    def store_feature_distribution(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_distribution,
        bq_table_name="feature_distribution"
    ) -> None:
        self._store_df(
            score_type, dataset_type, version_type, version, df_distribution, bq_table_name,
            [
                bigquery.SchemaField("feature_name", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("segment", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("count", 'INTEGER', 'REQUIRED'),
                bigquery.SchemaField("total", 'INTEGER', 'REQUIRED'),
                bigquery.SchemaField("percent", 'FLOAT', 'REQUIRED'),
            ]
        )

    def calc_score_distribution(
            self,
            score: np.ndarray,
            bins=[
                0, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125,
                0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
            ]):

        total = len(score)
        histogram = np.histogram(score, bins)[0]
        percent = histogram / total
        segments = self._cvt_bins2labels(bins)
        segments = [". ".join(e) for e in zip(str_utils.UPPER_ALPHABET, segments)]
        return pd.DataFrame({
            "segment": segments,
            "count": histogram,
            "total": [total] * len(histogram),
            "percent": percent,
        })
    
    def store_score_distribution(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_distribution: pd.DataFrame,
        bq_table_name="score_distribution"
    ) -> None:
        self._store_df(
            score_type, dataset_type, version_type, version, df_distribution, bq_table_name,
            [
                bigquery.SchemaField("segment", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("count", 'INTEGER', 'REQUIRED'),
                bigquery.SchemaField("total", 'INTEGER', 'REQUIRED'),
                bigquery.SchemaField("percent", 'FLOAT', 'REQUIRED'),
            ]
        )
    
    def _check_norm(self, df: pd.DataFrame):
        try:
            if df.__is_norm == True:
                return True
        except Exception as e:
            return False

    def _clear_data(
        self,
        full_table_id: str,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
    ) -> None:
        try:
            self.bq_client.query(f"""
                DELETE FROM {full_table_id}
                WHERE score_type = '{score_type}'
                AND dataset_type = '{dataset_type}'
                AND version_type = '{version_type}'
                AND version = '{version}'
            """).result()
        except NotFound as e:
            print(f"'{full_table_id}' is not found.")

    def _store_df(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df: pd.DataFrame,
        bq_table_name: str,
        schema: List[bigquery.SchemaField]
    ):
        job_config = bigquery.LoadJobConfig(
            schema=[
                *self.ID_SCHEMA,
                *schema,
                bigquery.SchemaField("utc_update_at", "DATETIME", "REQUIRED")
            ],
            clustering_fields=self.ID_COLS,
        )
        table_id = f"{self.project}.{self.dataset}.{bq_table_name}"
        df = df.copy()
        df["score_type"] = score_type
        df["dataset_type"] = dataset_type
        df["version_type"] = version_type
        df["version"] = version
        df["utc_update_at"] = datetime.now()
        self._clear_data(table_id, score_type, dataset_type, version_type, version)
        return self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config).result()

    def _cvt_bins2labels(self, bins: List[object]) -> List[str]:
        if len(bins) <= 1:
            return []
        bins = [round(float(e), 2) for e in bins]
        bins = [int(e) if e.is_integer() else e for e in bins]
        segments = ["[" + ", ".join(map(str, e)) + ")" for e in zip(bins[:-1], bins[1:])]
        segments[-1] = segments[-1][:-1] + "]"
        return segments
