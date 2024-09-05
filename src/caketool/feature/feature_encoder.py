import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from caketool.utils.lib_utils import get_class
from category_encoders.utils import BaseEncoder

class FeatureEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, encoder_name, **args) -> None:
        self.encoder_name = encoder_name
        self.encoder_class = get_class(encoder_name)
        self.encoder: BaseEncoder = self.encoder_class(**args)
    
    def fit(self, X: pd.DataFrame, y=None):
        object_cols = list(X.select_dtypes(['object']).columns)
        if len(object_cols) == 0:
            return self
        self.encoder.fit(X[object_cols], y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder is None or self.encoder.cols is None:
            return X
        X = X.copy()
        object_cols = list(X.select_dtypes(['object']).columns)
        X[object_cols] = self.encoder.transform(X[object_cols])
        return X