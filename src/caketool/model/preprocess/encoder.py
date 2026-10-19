import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from caketool.utils.lib_utils import get_class

try:
    from category_encoders.utils import SupervisedTransformerMixin as _SupervisedEncoder
except ImportError:  # pragma: no cover - older category_encoders
    _SupervisedEncoder = None

def _categorical_columns(X: pd.DataFrame) -> list[str]:
    """Return object, string and category columns.

    pandas 3 stores text as the ``str`` dtype instead of ``object``, so
    ``select_dtypes("object")`` alone is not enough.
    """
    return [
        col
        for col, dtype in X.dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    ]


_SUPERVISED_ENCODER_NAMES = {
    "TargetEncoder",
    "LeaveOneOutEncoder",
    "JamesSteinEncoder",
    "MEstimateEncoder",
    "WOEEncoder",
    "CatBoostEncoder",
    "GLMMEncoder",
    "QuantileEncoder",
    "SummaryEncoder",
}


def _is_supervised(encoder) -> bool:
    if _SupervisedEncoder is not None and isinstance(encoder, _SupervisedEncoder):
        return True
    return type(encoder).__name__ in _SUPERVISED_ENCODER_NAMES


class FeatureEncoder(TransformerMixin, BaseEstimator):
    """A wrapper for categorical encoders that automatically detects and encodes object columns.

    This encoder wraps any encoder from category_encoders or similar libraries that
    follow the sklearn API and have a ``cols`` attribute after fitting.

    Parameters
    ----------
    encoder_name : str
        Fully qualified class name of the encoder (e.g., ``"category_encoders.TargetEncoder"``).
    cv : int or None, optional
        Number of folds for out-of-fold encoding in ``fit_transform``. When set
        and the encoder uses the target, each training row is encoded by an
        encoder fitted on the other folds, so downstream steps never see the
        row's own target. ``transform`` always uses the encoder fitted on all
        rows. Default ``None`` (no cross-fitting).
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
    feature_lineage_ : dict[str, list[str]]
        Maps each encoded output column to the input column(s) it came from.

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

    def __init__(self, encoder_name: str, cv: int | None = None, **kwargs) -> None:
        self.encoder_name = encoder_name
        self.cv = cv
        self.encoder_kwargs = kwargs

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn clone() compatibility)."""
        params = {"encoder_name": self.encoder_name, "cv": self.cv}
        params.update(self.encoder_kwargs)
        return params

    def set_params(self, **params):
        """Set parameters for this estimator (sklearn clone() compatibility)."""
        if "encoder_name" in params:
            self.encoder_name = params.pop("encoder_name")
        if "cv" in params:
            self.cv = params.pop("cv")
        self.encoder_kwargs.update(params)
        return self

    def _new_encoder(self):
        return get_class(self.encoder_name)(**self.encoder_kwargs)

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
        self.encoder_ = self._new_encoder()

        self.object_cols_ = _categorical_columns(X)
        if len(self.object_cols_) == 0:
            self.fitted_cols_ = []
            self.feature_lineage_ = {}
            return self
        y_aligned = None if y is None else pd.Series(np.asarray(y), index=X.index)
        self.encoder_.fit(X[self.object_cols_], y_aligned)
        self.fitted_cols_ = self.encoder_.cols or []
        self.feature_lineage_ = self._build_lineage(X)
        return self

    def _build_lineage(self, X: pd.DataFrame) -> dict[str, list[str]]:
        """Map encoded output columns back to the input columns they came from."""
        if not self.fitted_cols_:
            return {}
        output_cols = list(self.encoder_.transform(X[self.fitted_cols_].iloc[:1]).columns)
        lineage: dict[str, list[str]] = {}
        for out_col in output_cols:
            if out_col in self.fitted_cols_:
                lineage[out_col] = [out_col]
                continue
            # Multi-column encoders (OneHot, Binary, BaseN, ...) name outputs "<col>_<suffix>".
            prefixes = [c for c in self.fitted_cols_ if str(out_col).startswith(f"{c}_")]
            lineage[out_col] = [max(prefixes, key=len)] if prefixes else list(self.fitted_cols_)
        return lineage

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit the encoder and return the encoded training data.

        With ``cv`` set and a target-based encoder, training rows are encoded
        out-of-fold to avoid target leakage.
        """
        X_out = self.fit(X, y).transform(X)
        if not self.cv or y is None or not self.fitted_cols_ or not _is_supervised(self.encoder_):
            return X_out

        y_arr = np.asarray(y)
        _, counts = np.unique(y_arr, return_counts=True)
        if counts.min() >= self.cv:
            splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=0)
        else:
            splitter = KFold(n_splits=self.cv, shuffle=True, random_state=0)

        X_obj = X[self.fitted_cols_]
        encoded_cols = [c for c in self.feature_lineage_ if c in X_out.columns]
        oof = X_out[encoded_cols].to_numpy(dtype=float, copy=True)
        for train_idx, val_idx in splitter.split(X_obj, y_arr):
            fold_encoder = self._new_encoder()
            X_train = X_obj.iloc[train_idx]
            fold_encoder.fit(X_train, pd.Series(y_arr[train_idx], index=X_train.index))
            oof[val_idx] = fold_encoder.transform(X_obj.iloc[val_idx])[encoded_cols].to_numpy(dtype=float)
        X_out[encoded_cols] = oof
        return X_out

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

        current_cat_cols = set(_categorical_columns(X))
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

        # Encoders that expand a column (e.g. OneHot) do not return it under its own name.
        replaced = fitted_set - set(X_encoded.columns)
        final_order = [col for col in original_columns if col in X.columns and col not in replaced] + new_cols
        return X[final_order]
