"""Abstract base class for model explainers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ModelExplainer(ABC):
    """Abstract base class defining the interface for model explainers.

    All explainer implementations must provide fit, get_feature_importance,
    and get_local_explanation.
    """

    @abstractmethod
    def fit(self, X) -> ModelExplainer:
        """Compute explanations for the given input data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input data to explain.

        Returns
        -------
        self : ModelExplainer
        """

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Return global feature importance.

        Returns
        -------
        pd.DataFrame
            Feature importance table sorted by importance descending.
        """

    @abstractmethod
    def get_local_explanation(self, row_index: int = 0) -> pd.DataFrame:
        """Return explanation for a single sample.

        Parameters
        ----------
        row_index : int, optional
            Index of the sample to explain. Defaults to 0.

        Returns
        -------
        pd.DataFrame
            Per-sample explanation table sorted by importance descending.
        """
