import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .feature_remover import FeatureRemover


class ReduceVIF(FeatureRemover):
    def __init__(self, thresh=10):
        super().__init__([])
        self.thresh = thresh

    def fit(self, X: pd.DataFrame, y=None):
        dropped = True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]

            max_vif = max(vif)
            if max_vif > self.thresh:
                maxloc = vif.index(max_vif)
                droped_col = X.columns.tolist()[maxloc]
                X = X.drop([droped_col], axis=1)
                self.droped_cols.append(droped_col)
                dropped = True
        return self
