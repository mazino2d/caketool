from typing import List
from numbers import Number
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class InfinityHandler(TransformerMixin, BaseEstimator):
    def __init__(self, cols: List[str], def_val: Number = -100):
        self.cols = cols
        self.def_val = def_val

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in list(set(X[self.cols].columns.to_series()[np.isinf(X[self.cols]).any()])):
            X[col] = X[col].apply(lambda x: -100 if x == np.inf else x)
        return X
