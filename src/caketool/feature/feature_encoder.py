import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from caketool.utils.lib_utils import get_class

class FeatureEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, encoder_name, **args) -> None:
        self.encoder_name = encoder_name
        self.encoder_class = get_class(encoder_name)
        self.encoder: BaseEstimator = self.encoder_class(**args)
    
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
        # Check encode cols
        columns_to_encode = [col for col in self.encoder.cols if col in X.columns]
        if not columns_to_encode:
            return X
        # Fill missing cols
        missing_columns = [col for col in self.encoder.cols if col not in X.columns]
        if missing_columns:
            X[missing_columns] = np.nan
        # Encode cols
        X_encoded: pd.DataFrame = self.encoder.transform(X[self.encoder.cols])
        X_encoded = X_encoded.drop(columns=missing_columns)
        # Fill non-encode cols
        non_encoded_columns = [col for col in X.columns if col not in self.encoder.cols]
        X_final = pd.concat([X[non_encoded_columns], X_encoded], axis=1)

        return X_final
       