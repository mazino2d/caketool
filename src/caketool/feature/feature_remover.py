from typing import List
import numpy as np
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


class ColinearFeatureRemover(FeatureRemover):
    def __init__(self, correlation_threshold=0.9):
        super().__init__([])
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        correlations = []
        for col in X.columns:
            correlations.append(np.abs(y.corr(X[col])))
        df_clusters = pd.DataFrame(
            zip(X.columns, correlations),
            columns=['feature', 'correlation']
        )
        df_clusters = df_clusters\
            .sort_values(by='correlation', ascending=False)\
            .reset_index(drop=True)
        df_clusters = df_clusters[~df_clusters["correlation"].isna()]
        to_remove_list = []
        corr = X[df_clusters['feature']].corr()

        for idx, col_a in enumerate(corr.columns):
            if col_a not in to_remove_list:
                for col_b in corr.columns[idx+1:]:
                    if corr[col_a][col_b] > self.correlation_threshold:
                        to_remove_list.append(col_b)

        self.droped_cols = to_remove_list
        return self
