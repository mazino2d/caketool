from numbers import Number
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class InfinityHandler(TransformerMixin, BaseEstimator):
    """
    A transformer that handles infinite values in specified columns by replacing 
    them with a default value.

    Parameters
    ----------
    cols : List[str]
        List of column names to check for infinite values.

    def_val : Number, optional (default=-100)
        The value to replace infinite values with.
    """

    def __init__(self, def_val: Number = -100):
        """
        Initialize the InfinityHandler with the specified columns and default value.

        Parameters
        ----------
        cols : List[str]
            List of column names to check for infinite values.

        def_val : Number, optional (default=-100)
            The value to replace infinite values with.
        """
        self.def_val = def_val

    def fit(self, X, y=None):
        """
        Fit the transformer. This method does not perform any fitting and is included 
        to maintain compatibility with scikit-learn's interface.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by replacing infinite values in the specified 
        columns with the default value.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        X : pd.DataFrame
            The transformed DataFrame with infinite values replaced.
        """
        columns = list(set(X.columns).difference(X.select_dtypes([object]).columns))
        for col in list(set(X[columns].columns.to_series()[np.isinf(X[columns]).any()])):
            X[col] = X[col].apply(lambda x: self.def_val if x == np.inf else x)
        return X
