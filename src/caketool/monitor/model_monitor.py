import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import bigquery


class ModelMonitor:

    def __init__(self, project: str, location: str, dataset: str = "mlops_motinor") -> None:
        self.project = project
        self.location = location
        self.dataset = dataset
        self.bq_client = bigquery.Client(project=self.project, location=self.location)

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
        return pd.DataFrame({
            "segment": list(map(str, bins[1:])),
            "count": histogram,
            "total": [total] * len(histogram),
            "percent": percent,
        })

    def store_distribution_looker(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_distribution: pd.DataFrame,
        bq_table_name="score_distribution_looker"
    ) -> pd.DataFrame:
        df_looker = df_distribution.copy()
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
