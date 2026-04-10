from numbers import Number

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InfinityHandler(TransformerMixin, BaseEstimator):
    """Replace infinite values in numeric columns with a fixed default value.

    Parameters
    ----------
    def_val : Number, optional
        Value to replace both ``+inf`` and ``-inf`` with. Default ``-100``.
    """

    def __init__(self, def_val: Number = -100):
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
        num_cols = X.select_dtypes(exclude="object").columns
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], self.def_val)
        return X
