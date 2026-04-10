from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MissingValueImputer(TransformerMixin, BaseEstimator):
    """Fill NaN values in numeric columns using a fitted imputation strategy.

    Designed to run after encoding steps that may produce NaN for unseen
    categories (e.g. TargetEncoder on a category not present in training).

    Parameters
    ----------
    strategy : {"median", "mean", "constant"}
        Imputation strategy. Default ``"median"``.
    fill_value : float
        Replacement value when ``strategy="constant"``. Default ``-999``
        (credit risk convention for missing/unknown).

    Attributes
    ----------
    fill_values_ : dict[str, float]
        Per-column fill values computed during fit (available after fit).
    """

    def __init__(self, strategy: Literal["median", "mean", "constant"] = "median", fill_value: float = -999):
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None) -> "MissingValueImputer":
        """Compute per-column fill values from training data.

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        num_cols = X.select_dtypes(exclude="object").columns
        if self.strategy == "median":
            self.fill_values_ = X[num_cols].median().to_dict()
        elif self.strategy == "mean":
            self.fill_values_ = X[num_cols].mean().to_dict()
        else:
            self.fill_values_ = {col: self.fill_value for col in num_cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in numeric columns using fitted fill values.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` with NaN values imputed.
        """
        check_is_fitted(self, "fill_values_")
        X = X.copy()
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X
