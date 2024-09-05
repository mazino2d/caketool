from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif


class FeatureRemover(TransformerMixin, BaseEstimator):
    """
    Transformer that removes specified columns from a DataFrame.

    Parameters
    ----------
    droped_cols : List[str], optional (default=[])
        List of column names to be removed.
    """

    def __init__(self, droped_cols: List[str] = []):
        """
        Initialize the FeatureRemover with the specified columns to be removed.

        Parameters
        ----------
        droped_cols : List[str], optional (default=[])
            List of column names to be removed.
        """
        self.droped_cols = droped_cols

    def fit(self, X, y=None):
        """
        Fit the transformer. This method does not perform any fitting and is included 
        to maintain compatibility with scikit-learn's interface.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        y : array-like, shape (n_samples,), optional (default=None)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame by removing the specified columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        Returns
        -------
        X : pd.DataFrame
            The transformed DataFrame with specified columns removed.
        """
        columns = list(set(self.droped_cols).intersection(X.columns))
        return X.drop(columns=columns)


class ColinearFeatureRemover(FeatureRemover):
    """
    Transformer that removes collinear features from a DataFrame based on a correlation threshold.

    Parameters
    ----------
    correlation_threshold : float, optional (default=0.9)
        The correlation threshold above which features are considered collinear and are removed.
    """

    def __init__(self, correlation_threshold=0.9):
        """
        Initialize the ColinearFeatureRemover with the specified correlation threshold.

        Parameters
        ----------
        correlation_threshold : float, optional (default=0.9)
            The correlation threshold above which features are considered collinear and are removed.
        """
        super().__init__([])
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the transformer by identifying collinear features to be removed.

        Parameters
        ----------
        X : pd.DataFrame
            The input samples.

        y : pd.Series, optional (default=None)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
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


class UnivariateFeatureRemover(FeatureRemover):
    def __init__(self, score_func: callable = f_classif, threshold=0.05):
        super().__init__([])
        self.score_func = score_func
        self.threshold = threshold
        self.feature_importance = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        f_statistic, p_values = self.score_func(X, y)
        self.feature_importance = pd.DataFrame({
            "features": X.columns,
            "f_statistic": f_statistic,
            "p_values": p_values
        }).fillna({
            "f_statistic": 0,
            "p_values": 1,
        })
        self.droped_cols = list(self.feature_importance[self.feature_importance["p_values"] > self.threshold]["features"])
        return self
