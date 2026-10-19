from numbers import Number

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InfinityHandler(TransformerMixin, BaseEstimator):
    """Replace infinite values in numeric columns.

    By default ``+inf`` and ``-inf`` become ``NaN``, which XGBoost treats as
    missing. Mapping both to one finite number would put ``+inf`` below every
    real value, so a finite replacement should only be used deliberately.

    Parameters
    ----------
    def_val : Number, optional
        Value to replace both ``+inf`` and ``-inf`` with. Default ``np.nan``.
    """

    def __init__(self, def_val: Number = np.nan):
        self.def_val = def_val

    def fit(self, X, y=None):
        """No-op fit for sklearn pipeline compatibility.

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace ``+inf`` and ``-inf`` in numeric columns with ``def_val``.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy of ``X`` with infinite values replaced.
        """
        X = X.copy()
        num_cols = X.select_dtypes("number").columns
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], self.def_val)
        return X
