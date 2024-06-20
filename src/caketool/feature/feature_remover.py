from typing import List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureRemover(TransformerMixin, BaseEstimator):
    def __init__(self, droped_cols: List[str] = []):
        self.droped_cols = droped_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        columns = list(set(self.droped_cols).intersection(X.columns))
        return X.drop(columns=columns)
