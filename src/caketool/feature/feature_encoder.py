import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from caketool.utils.lib_utils import get_class
from category_encoders.utils import BaseEncoder

class FeatureEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, encoder_name, **args) -> None:
        self.encoder_name = encoder_name
        self.encoder_class = get_class(encoder_name)
        self.encoder: BaseEncoder = self.encoder_class(**args)
    
    def fit(self, X, y=None):
        self.encoder.fit(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        object_cols = X.select_dtypes(['object']).columns
        if len(object_cols) == 0:
            return X
        X_enc = self.encoder.transform(X[object_cols])
        X = X.drop(columns=object_cols)
        return pd.concat([X, X_enc], axis = 1)
