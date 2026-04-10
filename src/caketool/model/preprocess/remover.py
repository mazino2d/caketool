import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class FeatureRemover(TransformerMixin, BaseEstimator):
    """Transformer that removes specified columns from a DataFrame.

    Parameters
    ----------
    dropped_cols : tuple[str, ...], optional
        Tuple of column names to be removed. Default ``()``.
    """

    def __init__(self, dropped_cols: tuple[str, ...] = ()):
        self.dropped_cols = dropped_cols

    def fit(self, X, y=None):
        """No-op fit for sklearn pipeline compatibility.

        Parameters
        ----------
        X : pd.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove the specified columns from the DataFrame.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        columns = list(set(self.dropped_cols).intersection(X.columns))
        return X.drop(columns=columns)


class ColinearFeatureRemover(FeatureRemover):
    """Remove collinear features based on a pairwise correlation threshold.

    Features are ranked by their absolute correlation with the target ``y``.
    Starting from the most correlated feature, any lower-ranked feature that
    is highly correlated (|r| > threshold) with an already-kept feature is
    removed.

    Parameters
    ----------
    correlation_threshold : float, optional
        Absolute correlation threshold above which a feature pair is considered
        collinear. Default ``0.9``.

    Attributes
    ----------
    dropped_cols : list[str]
        Names of removed features (available after fit).
    """

    def __init__(self, correlation_threshold: float = 0.9):
        super().__init__(())
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Identify collinear features.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
            Target values used to rank features by relevance.

        Returns
        -------
        self
        """
        correlations = [np.abs(y.corr(X[col])) for col in X.columns]
        df_clusters = pd.DataFrame(zip(X.columns, correlations, strict=True), columns=["feature", "correlation"])
        df_clusters = df_clusters.sort_values(by="correlation", ascending=False).reset_index(drop=True)
        df_clusters = df_clusters[~df_clusters["correlation"].isna()]

        corr = X[df_clusters["feature"]].corr()
        to_remove: list[str] = []

        for idx, col_a in enumerate(corr.columns):
            if col_a not in to_remove:
                for col_b in corr.columns[idx + 1 :]:
                    if abs(corr[col_a][col_b]) > self.correlation_threshold:
                        to_remove.append(col_b)

        self.dropped_cols = to_remove
        return self


class UnivariateFeatureRemover(FeatureRemover):
    """Remove features with low univariate statistical relevance.

    Uses a scoring function (default: F-test for classification) to compute a
    p-value per feature. Features whose p-value exceeds the threshold are dropped.

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
    feature_importance : pd.DataFrame or None
        DataFrame with columns ``['features', 'f_statistic', 'p_values']``
        after fit.
    dropped_cols : list[str]
        Names of columns identified for removal (available after fit).
    """

    def __init__(self, score_func: callable = f_classif, threshold: float = 0.05) -> None:
        super().__init__([])
        self.score_func = score_func
        self.threshold = threshold
        self.feature_importance = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "UnivariateFeatureRemover":
        """Identify low-relevance features via univariate statistical testing.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.Series
            Target labels required by ``score_func``.

        Returns
        -------
        self
        """
        f_statistic, p_values = self.score_func(X, y)
        self.feature_importance = pd.DataFrame(
            {"features": X.columns, "f_statistic": f_statistic, "p_values": p_values}
        ).fillna({"f_statistic": 0, "p_values": 1})
        self.dropped_cols = list(
            self.feature_importance[self.feature_importance["p_values"] > self.threshold]["features"]
        )
        return self
