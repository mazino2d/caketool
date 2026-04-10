import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class OutlierClipper(TransformerMixin, BaseEstimator):
    """Clip numeric features to quantile-based bounds (winsorization).

    Standard practice in credit risk modeling to limit the influence of
    extreme values before encoding and feature selection.

    Parameters
    ----------
    lower_quantile : float, optional
        Lower bound percentile. Values below this quantile are clipped.
        Default ``0.01``.
    upper_quantile : float, optional
        Upper bound percentile. Values above this quantile are clipped.
        Default ``0.99``.

    Attributes
    ----------
    bounds_ : dict[str, tuple[float, float]]
        Per-column ``(lower, upper)`` bounds computed during fit.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierClipper":
        """Compute per-column quantile bounds from training data.

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        num_cols = X.select_dtypes(exclude="object").columns
        self.bounds_ = {
            col: (X[col].quantile(self.lower_quantile), X[col].quantile(self.upper_quantile)) for col in num_cols
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip numeric columns to their fitted quantile bounds.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` with clipped values.
        """
        check_is_fitted(self, "bounds_")
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=lower, upper=upper)
        return X
