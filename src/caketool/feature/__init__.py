"""Window-based feature engineering for tabular event data.

Provides ``generate_features_by_window``, which aggregates raw event
records into per-entity feature vectors over configurable time windows.

Supported backends:

- **pandas** – Pure pandas; no extra install required.
- **polars** – Fast Rust-backed DataFrame library (``pip install polars``).
- **spark** – Apache Spark for large-scale distributed processing
  (``pip install pyspark``).
- **bigframes** – Google BigQuery DataFrames for serverless computation
  (``pip install bigframes``).

Example
-------
>>> from caketool.feature import generate_features_by_window
>>> features = generate_features_by_window(
...     df,
...     client_id_col="user_id",
...     lookback_days=(0, 7, 30),
...     numeric_cols=("amount", "quantity"),
...     backend="pandas",
... )
"""

from .feature_generator import (
    generate_features_by_window as generate_features_by_window,
)
