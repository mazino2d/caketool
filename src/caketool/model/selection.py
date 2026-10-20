from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_is_fitted


def _as_target(y, index: pd.Index) -> pd.Series:
    """Return ``y`` as a Series aligned positionally to ``index``."""
    return pd.Series(np.asarray(y), index=index)


class _FeatureSelector(TransformerMixin, BaseEstimator):
    """Base for selectors that learn ``dropped_cols_`` during fit and drop them in transform."""

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "dropped_cols_")

    def _set_selection(self, X: pd.DataFrame, dropped_cols: list[str]) -> None:
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.dropped_cols_ = dropped_cols
        dropped = set(dropped_cols)
        self.selected_cols_ = [col for col in X.columns if col not in dropped]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop the columns identified during fit.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            ``X`` without ``dropped_cols_``. Columns unseen during fit are kept.
        """
        check_is_fitted(self, "dropped_cols_")
        dropped = set(self.dropped_cols_)
        return X.drop(columns=[col for col in X.columns if col in dropped])

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the names of the selected features.

        Parameters
        ----------
        input_features : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        np.ndarray
            ``selected_cols_`` as an object array.
        """
        check_is_fitted(self, "dropped_cols_")
        return np.asarray(self.selected_cols_, dtype=object)


class ColinearFeatureSelector(_FeatureSelector):
    """Select features by removing collinear ones based on a pairwise correlation threshold.

    Features are ranked by their absolute correlation with the target ``y``.
    Starting from the most correlated feature, any lower-ranked feature that
    is highly correlated (|r| > threshold) with an already-kept feature is
    removed. Correlations are computed pairwise, ignoring missing values.

    Parameters
    ----------
    correlation_threshold : float, optional
        Absolute correlation threshold above which a feature pair is considered
        collinear. Default ``0.9``.

    Attributes
    ----------
    dropped_cols_ : list[str]
        Names of removed features (available after fit).
    selected_cols_ : list[str]
        Names of kept features, in input order (available after fit).

    Examples
    --------
    >>> selector = ColinearFeatureSelector(correlation_threshold=0.9).fit(X_train, y_train)
    >>> X_train_selected = selector.transform(X_train)
    """

    def __init__(self, correlation_threshold: float = 0.9):
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y=None) -> "ColinearFeatureSelector":
        """Identify collinear features.

        Parameters
        ----------
        X : pd.DataFrame
            Numeric features.
        y : array-like
            Target values used to rank features by relevance. Matched to ``X``
            by position, not by index.

        Returns
        -------
        self
        """
        correlations = X.corrwith(_as_target(y, X.index)).abs()
        ranked = correlations.dropna().sort_values(ascending=False).index

        corr = X[ranked].corr()
        to_remove: list[str] = []

        for idx, col_a in enumerate(corr.columns):
            if col_a not in to_remove:
                for col_b in corr.columns[idx + 1 :]:
                    if col_b not in to_remove and abs(corr[col_a][col_b]) > self.correlation_threshold:
                        to_remove.append(col_b)

        self._set_selection(X, to_remove)
        return self


class UnivariateFeatureSelector(_FeatureSelector):
    """Select features with univariate statistical relevance to the target.

    Uses a scoring function (default: F-test for classification) to compute a
    p-value per feature. Features whose p-value exceeds the threshold are dropped.
    Columns containing missing values are scored on their non-missing rows only.

    Parameters
    ----------
    score_func : callable, optional
        Univariate scoring function following the sklearn API: takes ``(X, y)``
        and returns ``(scores, p_values)``. Common choices: ``f_classif``,
        ``chi2``, ``f_regression``. Default ``f_classif``.
    threshold : float, optional
        Maximum allowed p-value. Features with ``p_value > threshold`` are
        removed. Default ``0.05``.

    Attributes
    ----------
    feature_importance_ : pd.DataFrame
        DataFrame with columns ``['features', 'f_statistic', 'p_values']``
        after fit.
    dropped_cols_ : list[str]
        Names of removed features (available after fit).
    selected_cols_ : list[str]
        Names of kept features, in input order (available after fit).

    Examples
    --------
    >>> selector = UnivariateFeatureSelector(threshold=0.05).fit(X_train, y_train)
    >>> selector.feature_importance_.sort_values("p_values").head()
    >>> X_train_selected = selector.transform(X_train)
    """

    def __init__(self, score_func: Callable = f_classif, threshold: float = 0.05) -> None:
        self.score_func = score_func
        self.threshold = threshold

    def _score(self, X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        has_nan = X.isna().any()
        if not has_nan.any():
            return self.score_func(X, y)

        f_statistic = np.full(X.shape[1], np.nan)
        p_values = np.full(X.shape[1], np.nan)
        for i, col in enumerate(X.columns):
            mask = X[col].notna().to_numpy()
            if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                continue
            f_stat, p_val = self.score_func(X.loc[mask, [col]], y[mask])
            f_statistic[i], p_values[i] = f_stat[0], p_val[0]
        return f_statistic, p_values

    def fit(self, X: pd.DataFrame, y=None) -> "UnivariateFeatureSelector":
        """Identify low-relevance features via univariate statistical testing.

        Parameters
        ----------
        X : pd.DataFrame
            Numeric features.
        y : array-like
            Target values required by ``score_func``, matched to ``X`` by position.

        Returns
        -------
        self
        """
        f_statistic, p_values = self._score(X, np.asarray(y))
        self.feature_importance_ = pd.DataFrame(
            {"features": X.columns, "f_statistic": f_statistic, "p_values": p_values}
        ).fillna({"f_statistic": 0, "p_values": 1})
        dropped = self.feature_importance_.loc[self.feature_importance_["p_values"] > self.threshold, "features"]
        self._set_selection(X, list(dropped))
        return self
