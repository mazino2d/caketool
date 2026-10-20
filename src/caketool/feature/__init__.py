"""Window-based feature engineering for tabular event data.

Provides ``generate_features_by_window``, which aggregates raw event
records (pandas) into per-entity feature vectors over configurable time
windows. Every aggregation of a column's type is computed by default; pass
``aggs`` to pick the aggregations per column and keep the run light.

Example
-------
>>> from caketool.feature import generate_features_by_window
>>> features = generate_features_by_window(
...     df,
...     client_id_col="user_id",
...     lookback_days=(0, 7, 30),
...     numeric_cols=("amount", "quantity"),
...     aggs={"amount": ["sum", "max"], "quantity": ["avg"]},
... )
"""

from .feature_generator import (
    generate_features_by_window as generate_features_by_window,
)
