from datetime import datetime

import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from caketool.utils import arr_utils, num_utils, str_utils


class ModelMonitor:
    """BigQuery-backed monitor for tracking feature and score distributions.

    Stores reference distributions (bin thresholds and per-bin counts) in
    BigQuery tables and provides utilities to normalise data, create binning
    schemas, and calculate current distributions for comparison.

    The typical workflow is:

    1. ``normalize_data`` – Standardise dtypes and fill/cap missing/infinity
       values so that all downstream steps receive clean data.
    2. ``create_bin_data`` – Compute percentile-based bin edges for numeric
       features and top-K category lists for categorical features.
    3. ``store_bin_data`` – Persist the bin schema to BigQuery.
    4. ``calc_feature_distribution`` – Compute per-bin counts for a dataset
       using the stored bin schema.
    5. ``store_feature_distribution`` – Persist the distribution to BigQuery.
    6. Later, compare stored distributions using PSI or similar metrics.

    Parameters
    ----------
    project : str
        Google Cloud project ID used for BigQuery operations.
    location : str
        BigQuery dataset location (e.g. ``"US"`` or ``"asia-southeast1"``).
    dataset : str, optional
        BigQuery dataset name where monitoring tables are created.
        Defaults to ``"model_motinor"`` (note: intentional legacy spelling).

    Attributes
    ----------
    MISSING : str
        Sentinel string used to represent missing / null categories
        (``"cake.miss"``).
    OTHER : str
        Sentinel string used to represent out-of-vocabulary categories
        (``"cake.other"``).
    bq_client : bigquery.Client
        Authenticated BigQuery client.
    ID_COLS : list[str]
        Fixed identifier columns added to every monitoring table:
        ``["score_type", "dataset_type", "version_type", "version"]``.
    """

    def __init__(self, project, location, dataset="model_motinor") -> None:
        self.project = project
        self.location = location
        self.dataset = dataset
        self.MISSING = "cake.miss"
        self.OTHER = "cake.other"
        self.bq_client = bigquery.Client(project=self.project, location=self.location)
        self.ID_COLS = ["score_type", "dataset_type", "version_type", "version"]
        self.ID_SCHEMA = [bigquery.SchemaField(name, "STRING", "REQUIRED") for name in self.ID_COLS]

    def normalize_data(
        self, df: pd.DataFrame, inplace: bool = False, cate_missing_values: set[str] | None = None
    ) -> pd.DataFrame:
        """Standardise data types and encode missing / infinity values.

        Applies the following transformations in order:

        1. Fills ``NaN`` with ``-100`` (numeric sentinel).
        2. Coerces columns to numeric where possible; remaining columns are
           treated as categorical (converted to lowercase strings).
        3. Replaces ``+inf`` with ``-100`` in numeric columns.
        4. Casts numeric columns to ``int`` or ``float`` based on whether
           fractional values are present.
        5. Replaces any negative numeric value with ``-100``.
        6. Replaces categorical values in *cate_missing_values* with
           ``self.MISSING``; strips Vietnamese diacritics and lowercases
           all other categorical values.

        A ``__is_norm`` attribute is set on the returned DataFrame so that
        downstream methods can confirm normalisation has been applied.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input features.
        inplace : bool, optional
            If ``True``, modifies *df* in place and returns it.  If
            ``False`` (default), operates on a copy.
        cate_missing_values : set[str] or None, optional
            Set of string values that should be treated as missing for
            categorical columns.  Defaults to
            ``{"-1", "-100", "unknown", ""}``.

        Returns
        -------
        pd.DataFrame
            Normalised DataFrame with ``__is_norm = True`` set as an
            attribute.
        """
        if cate_missing_values is None:
            cate_missing_values = {"-1", "-100", "unknown", ""}
        if not inplace:
            df = df.copy()
        # Fill missing value
        df = df.fillna(-100)
        # Norm categorical feature type
        categorical_features: list[str] = []
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, KeyError):
                categorical_features.append(col)
        df[categorical_features] = df[categorical_features].apply(lambda x: x.astype(str).str.lower())

        # Find numerical features
        numerical_features: list[str] = df.select_dtypes([int, float]).columns
        # Handle infinity value
        for col in list(set(df[numerical_features].columns.to_series()[np.isinf(df[numerical_features]).any()])):
            df[col] = df[col].apply(lambda x: -100 if x == np.inf else x)
        # Norm float feature
        float_features: list[str] = []
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
                lambda x: (
                    self.MISSING if x in cate_missing_values else str_utils.remove_vn_diacritics(x).lower().strip()
                )
            )
        df.__is_norm = True
        return df

    def create_bin_data(self, df: pd.DataFrame, n_bins=10) -> dict[str, list[float | str]]:
        """Compute bin thresholds for all features in *df*.

        For **numeric** columns, generates ``n_bins`` percentile-based
        edges from the positive (non-missing) values.  A leading ``0.0``
        edge is always prepended so that the first bin captures the
        zero/missing mass, and the final edge is nudged up by ``1e-10``
        to make the last interval closed on the right.

        For **categorical** columns (object dtype), collects the top
        ``n_bins`` most frequent non-missing categories plus sentinel
        values ``self.MISSING`` and ``self.OTHER``.

        Parameters
        ----------
        df : pd.DataFrame
            Normalised DataFrame (must have been processed by
            ``normalize_data`` first).
        n_bins : int, optional
            Number of percentile bins for numeric features and maximum
            number of categories to retain for categorical features.
            Defaults to ``10``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``["feature_name", "type", "bins"]``:

            - ``feature_name`` – Column name.
            - ``type`` – Data type string (e.g. ``"float64"``, ``"int64"``,
              ``"string"``).
            - ``bins`` – List of numeric bin edges (float) for numeric
              features, or list of category strings for categorical features.

            Rows are sorted alphabetically by ``feature_name``.

        Raises
        ------
        Exception
            If *df* has not been normalised with ``normalize_data`` first.
        """
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
        bq_table_name="feature_bins",
    ):
        """Persist a bin-threshold table to BigQuery.

        Converts all bin values to strings before upload (BigQuery stores
        the ``bins`` column as a repeated ``STRING`` field so that numeric
        and categorical bins share the same schema).  Existing rows for the
        same ``(score_type, dataset_type, version_type, version)`` key are
        deleted before inserting the new data.

        Parameters
        ----------
        score_type : str
            Model or score identifier (e.g. ``"credit_score_v2"``).
        dataset_type : str
            Dataset split label (e.g. ``"train"``, ``"oot"``).
        version_type : str
            Version dimension (e.g. ``"champion"``, ``"challenger"``).
        version : str
            Specific version value (e.g. ``"2024-01"``).
        df_bins : pd.DataFrame
            Bin-threshold table produced by ``create_bin_data``.
        bq_table_name : str, optional
            Fully qualified or simple BigQuery table name.
            Defaults to ``"feature_bins"``.
        """
        df_bins["bins"] = df_bins["bins"].apply(lambda ls: list(map(str, ls)))
        self._store_df(
            score_type,
            dataset_type,
            version_type,
            version,
            df_bins,
            bq_table_name,
            [
                bigquery.SchemaField("feature_name", "STRING", "REQUIRED"),
                bigquery.SchemaField("type", "STRING", "REQUIRED"),
                bigquery.SchemaField("bins", "STRING", "REPEATED"),
            ],
        )

    def load_bin_data(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        bq_table_name: str,
    ):
        """Load a bin-threshold table from BigQuery.

        Fetches the bin schema previously stored by ``store_bin_data`` and
        restores numeric bin edges to ``float`` (categorical bins remain
        as string lists).

        Parameters
        ----------
        score_type : str
            Model or score identifier.
        dataset_type : str
            Dataset split label.
        version_type : str
            Version dimension.
        version : str
            Specific version value.
        bq_table_name : str
            Fully qualified BigQuery table name.

        Returns
        -------
        pd.DataFrame
            Bin-threshold DataFrame with columns
            ``["feature_name", "type", "bins"]``, sorted alphabetically
            by ``feature_name``.  Numeric bins are lists of ``float``;
            categorical bins are lists of ``str``.
        """
        df_bins: pd.DataFrame = self.bq_client.query(f"""
            SELECT feature_name, type, bins FROM {bq_table_name}
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

    def calc_feature_distribution(self, df: pd.DataFrame, df_bins: dict[str, np.ndarray]) -> pd.DataFrame:
        """Compute per-bin counts for each feature using the provided bin schema.

        For **numeric** features, uses ``np.histogram`` with ``[-inf, *bins, +inf]``
        boundaries so that out-of-range values are captured in the first
        (``"missing"``) and last (``"other"``) segments.

        For **categorical** features, maps any value not present in the
        stored category list to ``self.OTHER`` and counts occurrences per
        category.

        Segment labels follow the pattern ``"A. [lo, hi)"`` (numeric) or
        ``"A. category_name"`` (categorical), where the letter prefix
        (A–Z) allows consistent ordering when displayed in BI tools.

        Parameters
        ----------
        df : pd.DataFrame
            Normalised DataFrame (must have been processed by
            ``normalize_data`` first).
        df_bins : pd.DataFrame
            Bin-threshold table as returned by ``load_bin_data`` or
            ``create_bin_data``.

        Returns
        -------
        pd.DataFrame
            Exploded distribution table with columns:

            - ``feature_name`` – Feature column name.
            - ``segment`` – Labelled bin string (e.g. ``"A. [0, 100)"``).
            - ``count`` – Number of observations in this segment.
            - ``total`` – Total observations for this feature.
            - ``percent`` – Fraction of observations in this segment.

        Raises
        ------
        Exception
            If *df* has not been normalised with ``normalize_data`` first.
        """
        if not self._check_norm(df):
            raise Exception("DataFrame has not been normalized yet. Please use self.normalize_data(df)")
        bin_thresholds: dict[str, np.ndarray] = dict(zip(df_bins.feature_name, df_bins.bins, strict=True))
        hists = []
        categorical_features = [k for k, v in bin_thresholds.items() if len(v) > 0 and isinstance(v[0], str)]
        numerical_features = [k for k, v in bin_thresholds.items() if len(v) > 0 and not isinstance(v[0], str)]

        for f in numerical_features:
            series = df[f]
            hist, _ = np.histogram(series, [-np.inf, *bin_thresholds[f], np.inf])
            segments = ["missing", *self._cvt_bins2labels(bin_thresholds[f]), "other"]
            segments = [". ".join(e) for e in zip(str_utils.UPPER_ALPHABET, segments, strict=False)]
            hists.append([f, segments, hist, hist.sum(), hist / hist.sum()])
        for f in categorical_features:
            series = df[f]
            bins = bin_thresholds[f]
            series = series.apply(lambda x, b=bins: x if x in b else self.OTHER)
            vc = series.value_counts()
            for bin_name in bins:
                if bin_name not in vc.index:
                    vc.loc[bin_name] = 0
            vc = vc.reindex(bins)
            segments = [". ".join(e) for e in zip(str_utils.UPPER_ALPHABET, bins, strict=False)]
            hists.append([f, segments, vc, vc.sum(), vc / vc.sum()])

        return pd.DataFrame(hists, columns=["feature_name", "segment", "count", "total", "percent"]).explode(
            ["segment", "count", "percent"]
        )

    def store_feature_distribution(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_distribution,
        bq_table_name="feature_distribution",
    ) -> None:
        self._store_df(
            score_type,
            dataset_type,
            version_type,
            version,
            df_distribution,
            bq_table_name,
            [
                bigquery.SchemaField("feature_name", "STRING", "REQUIRED"),
                bigquery.SchemaField("segment", "STRING", "REQUIRED"),
                bigquery.SchemaField("count", "INTEGER", "REQUIRED"),
                bigquery.SchemaField("total", "INTEGER", "REQUIRED"),
                bigquery.SchemaField("percent", "FLOAT", "REQUIRED"),
            ],
        )

    def calc_score_distribution(self, score: np.ndarray, bins: int | list[float] = 10):
        """Compute the distribution of a score array over percentile-based bins.

        Parameters
        ----------
        score : np.ndarray
            1-D array of predicted scores (e.g. model probabilities).
        bins : int or list[float], optional
            If an integer, creates that many equal-frequency bins via
            ``create_percentile_bins``.  If a list of floats, uses them
            directly as bin edges.  Defaults to ``10``.

        Returns
        -------
        pd.DataFrame
            Distribution table with columns:

            - ``segment`` – Labelled interval string (e.g. ``"A. [0.0, 0.1)"``)
            - ``count`` – Number of scores in this bin.
            - ``total`` – Total number of scores.
            - ``percent`` – Fraction of scores in this bin.
        """
        if isinstance(bins, int):
            bins = arr_utils.create_percentile_bins(score, bins)
        total = len(score)
        histogram = np.histogram(score, bins)[0]
        percent = histogram / total
        segments = self._cvt_bins2labels(bins)
        segments = [". ".join(e) for e in zip(str_utils.UPPER_ALPHABET, segments, strict=False)]
        return pd.DataFrame(
            {
                "segment": segments,
                "count": histogram,
                "total": [total] * len(histogram),
                "percent": percent,
            }
        )

    def store_score_distribution(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df_distribution: pd.DataFrame,
        bq_table_name="score_distribution",
    ) -> None:
        self._store_df(
            score_type,
            dataset_type,
            version_type,
            version,
            df_distribution,
            bq_table_name,
            [
                bigquery.SchemaField("segment", "STRING", "REQUIRED"),
                bigquery.SchemaField("count", "INTEGER", "REQUIRED"),
                bigquery.SchemaField("total", "INTEGER", "REQUIRED"),
                bigquery.SchemaField("percent", "FLOAT", "REQUIRED"),
            ],
        )

    def _check_norm(self, df: pd.DataFrame):
        try:
            if df.__is_norm:
                return True
        except Exception:
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
        except NotFound:
            print(f"'{full_table_id}' is not found.")

    def _store_df(
        self,
        score_type: str,
        dataset_type: str,
        version_type: str,
        version: str,
        df: pd.DataFrame,
        bq_table_name: str,
        schema: list[bigquery.SchemaField],
    ):
        job_config = bigquery.LoadJobConfig(
            schema=[*self.ID_SCHEMA, *schema, bigquery.SchemaField("utc_update_at", "DATETIME", "REQUIRED")],
            clustering_fields=self.ID_COLS,
        )
        df = df.copy()
        df["score_type"] = score_type
        df["dataset_type"] = dataset_type
        df["version_type"] = version_type
        df["version"] = version
        df["utc_update_at"] = datetime.now()
        self._clear_data(bq_table_name, score_type, dataset_type, version_type, version)
        return self.bq_client.load_table_from_dataframe(df, bq_table_name, job_config=job_config).result()

    def _cvt_bins2labels(self, bins: list[object]) -> list[str]:
        if len(bins) <= 1:
            return []
        bins = [round(float(e), 2) for e in bins]
        bins = [int(e) if e.is_integer() else e for e in bins]
        segments = ["[" + ", ".join(map(str, e)) + ")" for e in zip(bins[:-1], bins[1:], strict=True)]
        segments[-1] = segments[-1][:-1] + "]"
        return segments
