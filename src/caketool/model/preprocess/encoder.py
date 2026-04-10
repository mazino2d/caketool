import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from caketool.utils.lib_utils import get_class


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """A wrapper for categorical encoders that automatically detects and encodes object columns.

    This encoder wraps any encoder from category_encoders or similar libraries that
    follow the sklearn API and have a ``cols`` attribute after fitting.

    Parameters
    ----------
    encoder_name : str
        Fully qualified class name of the encoder (e.g., ``"category_encoders.TargetEncoder"``).
    **kwargs
        Additional arguments passed to the encoder constructor.

    Attributes
    ----------
    encoder_ : BaseEstimator
        The fitted encoder instance (available after fit).
    object_cols_ : list[str]
        List of object/category columns detected during fit.
    fitted_cols_ : list[str]
        List of columns actually encoded.

    Supported Encoders
    ------------------
    Target-based (require y):
        - category_encoders.TargetEncoder: Mean target per category
        - category_encoders.LeaveOneOutEncoder: LOO mean, reduces overfitting
        - category_encoders.JamesSteinEncoder: Shrinkage toward global mean
        - category_encoders.MEstimateEncoder: Smoothed target encoding
        - category_encoders.WOEEncoder: Log odds ratio (binary target only)
        - category_encoders.CatBoostEncoder: Ordered target encoding

    Unsupervised (no y required):
        - category_encoders.OneHotEncoder: Binary columns per category
        - category_encoders.OrdinalEncoder: Integer encoding
        - category_encoders.BinaryEncoder: Binary representation
        - category_encoders.HashingEncoder: Hash trick, fixed dimensions
        - category_encoders.HelmertEncoder: Compare to mean of previous
        - category_encoders.SumEncoder: Deviation coding
        - category_encoders.BaseNEncoder: Base-N representation

    Examples
    --------
    >>> encoder = FeatureEncoder("category_encoders.TargetEncoder", smoothing=1.0)
    >>> encoder.fit(X_train, y_train)
    >>> X_encoded = encoder.transform(X_test)
    """

    def __init__(self, encoder_name: str, **kwargs) -> None:
        self.encoder_name = encoder_name
        self.encoder_kwargs = kwargs

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn clone() compatibility)."""
        params = {"encoder_name": self.encoder_name}
        params.update(self.encoder_kwargs)
        return params

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn clone() compatibility)."""
        if "encoder_name" in params:
            self.encoder_name = params.pop("encoder_name")
        self.encoder_kwargs.update(params)
        return self

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEncoder":
        """Fit the encoder on object columns.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : array-like, optional
            Target values (required for target-based encoders).

        Returns
        -------
        self : FeatureEncoder
        """
        encoder_class = get_class(self.encoder_name)
        self.encoder_ = encoder_class(**self.encoder_kwargs)

        self.object_cols_ = list(X.select_dtypes(["object", "category"]).columns)
        if len(self.object_cols_) == 0:
            self.fitted_cols_ = []
            return self
        self.encoder_.fit(X[self.object_cols_], y)
        self.fitted_cols_ = self.encoder_.cols or []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform object columns using the fitted encoder.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with encoded columns.
        """
        check_is_fitted(self, ["object_cols_", "fitted_cols_", "encoder_"])

        if not self.fitted_cols_:
            return X

        X = X.copy()
        original_columns = list(X.columns)
        original_set = set(original_columns)
        fitted_set = set(self.fitted_cols_)
        object_cols_set = set(self.object_cols_)

        current_cat_cols = set(X.select_dtypes(["object", "category"]).columns)
        new_cat_cols = current_cat_cols - object_cols_set
        if new_cat_cols:
            warnings.warn(
                f"New categorical columns {list(new_cat_cols)} found that were not in training data. "
                "They will not be encoded.",
                UserWarning,
                stacklevel=2,
            )

        columns_to_encode = fitted_set & original_set
        if not columns_to_encode:
            warnings.warn(
                f"None of the fitted columns {self.fitted_cols_} found in input. Returning unchanged DataFrame.",
                UserWarning,
                stacklevel=2,
            )
            return X

        missing_columns = fitted_set - original_set
        if missing_columns:
            warnings.warn(
                f"Columns {list(missing_columns)} were fitted but not found in input. "
                "They will be filled with NaN for encoding.",
                UserWarning,
                stacklevel=2,
            )
            for col in missing_columns:
                X[col] = np.nan

        X_encoded: pd.DataFrame = self.encoder_.transform(X[self.fitted_cols_])

        if missing_columns:
            X_encoded = X_encoded.drop(columns=list(missing_columns), errors="ignore")

        new_cols = [col for col in X_encoded.columns if col not in original_set]

        for col in X_encoded.columns:
            if col in original_set or col in new_cols:
                X[col] = X_encoded[col].values

        if missing_columns:
            X = X.drop(columns=list(missing_columns), errors="ignore")

        final_order = [col for col in original_columns if col in X.columns] + new_cols
        return X[final_order]
