from typing import List, Set, Dict, Union
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from caketool.text import strip_vietnamese_accents


class FeatureMonitor:

    def __init__(self, project, location, dataset="mlops_motinor") -> None:
        self.project = project
        self.location = location
        self.dataset = dataset
        self.MISSING = 'cake.miss'
        self.OTHER = 'cake.other'
        self.bq_client = bigquery.Client(project=self.project, location=self.location)

    def round(self, val: Union[int, float], type: type):
        if type == int:
            num_digits = len(str(int(val)))
            round_to = (num_digits - 2) * -1

            if num_digits > 2:
                return round(val, round_to)
            else:
                return int(val)
        elif type == float:
            return round(val, 2)
        else:
            pass
        return val

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
            df[col] = df[col].apply(lambda x: self.MISSING if x in cate_missing_values else strip_vietnamese_accents(x).lower().strip())

        return df

    def create_bin_data(
        self,
        df: pd.DataFrame,
        n_bins=10
    ) -> Dict[str, np.ndarray]:
        numerical_features = list(df.select_dtypes([int, float]).columns)
        int_features = list(df.select_dtypes([int]).columns)
        float_features = list(df.select_dtypes([float]).columns)
        categorical_features = list(df.select_dtypes([object]).columns)
        percentage = np.linspace(0, 100, n_bins + 1) / 100
        bin_thresholds = dict()
        for f in numerical_features:
            series = df[f]
            series = series[(series > 0) & (~series.isna())]
            if len(series) == 0:
                bin_thresholds[f] = []
                continue
            percentage = np.linspace(0, 100, n_bins + 1)
            bins = np.percentile(series, percentage)
            if f in int_features:
                bins = [self.round(e, int) for e in bins]
            if f in float_features:
                bins = [self.round(e, float) for e in bins]

            if bins[0] != 0:
                bins = [0, *bins]
            bins = np.unique(bins)
            if len(bins) >= 2:
                bins[-1] = bins[-1] + 1e-10
            bin_thresholds[f] = bins
        for f in categorical_features:
            series = df[f]
            bins = series.value_counts()[:n_bins].index.to_list()
            bins = sorted(set([*bins, self.MISSING, self.OTHER]))
            bin_thresholds[f] = bins

        return bin_thresholds

    def load_bin_data(
        self,
        dataset_id,
        table_name="feature_distribution"
    ) -> Dict[str, np.ndarray]:
        table_id = f"{self.project}.{self.dataset}.{table_name}"
        df_distribution: pd.DataFrame = self.bq_client.query(f"""
            SELECT feature_name, type, bins FROM {table_id}
            WHERE dataset_id = '{dataset_id}'
        """).to_dataframe()

        bin_thresholds = {}
        for _, row in df_distribution.iterrows():
            if row["type"] == "num":
                bin_thresholds[row["feature_name"]] = list(map(float, row["bins"]))
            if row["type"] == "cate":
                bin_thresholds[row["feature_name"]] = list(map(str, row["bins"]))

        return bin_thresholds

    def calculate_distribution(
        self,
        df: pd.DataFrame,
        bin_thresholds: Dict[str, np.ndarray]
    ):
        hists = []
        numerical_features = list(df.select_dtypes([int, float]).columns)
        categorical_features = list(df.select_dtypes([object]).columns)

        for f in numerical_features:
            series = df[f]
            hist, _ = np.histogram(series, [-np.inf, *bin_thresholds[f], np.inf])
            hists.append([f, "num", list(map(str, bin_thresholds[f])), hist.astype(np.int64).tolist()])
        for f in categorical_features:
            series = df[f]
            bins = bin_thresholds[f]
            series = series.apply(lambda x: x if x in bins else self.OTHER)
            vc = series.value_counts()
            for bin_name in bins:
                if bin_name not in vc.index:
                    vc.loc[bin_name] = 0
            vc = vc.reindex(bins)
            hists.append([f, "cate", vc.index.to_list(), vc.tolist()])

        return pd.DataFrame(hists, columns=["feature_name", "type", "bins", "histogram"])

    def store_distribution_bigquery(
            self,
            df_distribution: pd.DataFrame,
            dataset_id: str,
            bq_table_name="feature_distribution"
    ) -> None:
        df_distribution = df_distribution.copy()
        df_distribution["dataset_id"] = dataset_id
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("dataset_id", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("feature_name", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("type", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("bins", 'STRING', 'REPEATED'),
                bigquery.SchemaField("histogram", 'INTEGER', 'REPEATED'),
            ],
            clustering_fields=["dataset_id"]
        )
        table_id = f"{self.project}.{self.dataset}.{bq_table_name}"
        self.bq_client.query(f"DELETE FROM {table_id} WHERE dataset_id = '{dataset_id}'")
        self.bq_client.load_table_from_dataframe(df_distribution, table_id, job_config=job_config)

    def load_distribution(self, dataset_id: str, table_name: str = "feature_distribution") -> pd.DataFrame:
        table_id = f"{self.project}.{self.dataset}.{table_name}"
        df_distribution: pd.DataFrame = self.bq_client.query(f"""
            SELECT * FROM {table_id}
            WHERE dataset_id = '{dataset_id}'
        """).to_dataframe()
        return df_distribution

    def store_distribution_looker(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_distribution,
        bq_table_name="feature_distribution_looker"
    ) -> pd.DataFrame:
        ls_row = []
        for _, row in df_distribution.iterrows():
            feature_name = row["feature_name"]
            bins = row["bins"]
            histogram = row["histogram"]
            total = np.sum(histogram)
            if row["type"] == "cate":
                dist = zip(bins, histogram)
            if row["type"] == "num":
                bins = [round(float(e), 2) for e in row["bins"]]
                bins = [int(e) if e.is_integer() else e for e in bins]
                if len(bins) <= 0:
                    continue
                segment_label = []
                for s, e in zip(bins[:-1], bins[1:]):
                    segment_label.append(f"[{s}, {e})")
                if len(segment_label) > 0:
                    segment_label[-1] = segment_label[-1].replace(")", "]")
                segment_label = [f"(-inf, {bins[0]})", *segment_label, f"({bins[-1]}, +inf)"]
                dist = zip(segment_label, histogram)
            for k, v in dist:
                ls_row.append([
                    feature_name, k, v, total, v/total
                ])

        df_looker = pd.DataFrame(ls_row, columns=["feature_name", "segment", "count", "total", "percent"])
        df_looker["score_type"] = score_type
        df_looker["dataset_type"] = dataset_type
        df_looker["version_type"] = version_type
        df_looker["version"] = version
        df_looker["utc_update_at"] = datetime.now()

        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("score_type", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("dataset_type", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("version_type", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("version", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("feature_name", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("segment", 'STRING', 'REQUIRED'),
                bigquery.SchemaField("count", 'INTEGER', 'REQUIRED'),
                bigquery.SchemaField("total", 'INTEGER', 'REQUIRED'),
                bigquery.SchemaField("percent", 'FLOAT', 'REQUIRED'),
                bigquery.SchemaField("utc_update_at", 'DATETIME', 'REQUIRED'),
            ],
            clustering_fields=["score_type", "dataset_type", "version_type", "version"]
        )
        table_id = f"{self.project}.{self.dataset}.{bq_table_name}"
        self.bq_client.query(f"""
            DELETE FROM {table_id}
            WHERE score_type = '{score_type}'
            AND dataset_type = '{dataset_type}'
            AND version_type = '{version_type}'
            AND version = '{version}'
        """)
        self.bq_client.load_table_from_dataframe(df_looker, table_id, job_config=job_config)
        return df_looker

    def calculate_feature_drift_psi(self, dev_distribution: pd.DataFrame, test_distribution: pd.DataFrame) -> pd.DataFrame:
        comp_psi = pd.merge(
            dev_distribution, test_distribution,
            how="inner",
            on=["feature_name", "type"],
            suffixes=["_dev", "_test"]
        )

        for col in ["histogram_dev", "histogram_test"]:
            if comp_psi[col].apply(lambda s: s.sum()).nunique() != 1:
                raise ValueError(f"Sizing of {col} is not consistent between the features of the dataset!")

        wrong_bins_features = comp_psi[comp_psi.apply(lambda s: not (s["bins_dev"] == s["bins_test"]).all(), axis=1)]["feature_name"]
        if len(wrong_bins_features) > 0:
            raise ValueError(f"Bins is not the same between 2 datasets. {','.join(wrong_bins_features)}")

        def calc_psi(r: pd.Series) -> float:
            dev_percents: np.ndarray = r["histogram_dev"] / np.sum(r["histogram_dev"])
            test_percents: np.ndarray = r["histogram_test"] / np.sum(r["histogram_test"])
            dev_percents = np.clip(dev_percents, a_min=0.00001, a_max=None)
            test_percents = np.clip(test_percents, a_min=0.00001, a_max=None)
            if len(dev_percents) != len(test_percents):
                print(r)
            psi_value = (dev_percents - test_percents) * np.log(dev_percents / test_percents)
            psi_value = sum(psi_value)
            return psi_value

        comp_psi["psi"] = comp_psi.apply(calc_psi, axis=1)
        return comp_psi
