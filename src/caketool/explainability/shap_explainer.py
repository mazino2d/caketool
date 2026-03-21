"""SHAP-based model explainability for sklearn-compatible models."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from caketool.utils.lib_utils import require_dependencies

_NOT_FITTED_MSG = "ShapExplainer has not been fitted yet. Call .fit(X) first."
_VALID_EXPLAINER_TYPES = ("tree", "linear", "kernel")


class ShapExplainer(BaseEstimator):
    """
    SHAP-based model explainer supporting tree, linear, and kernel explainers.

    Parameters
    ----------
    model : object
        Fitted sklearn-compatible model.
    explainer_type : {"tree", "linear", "kernel"}, optional
        Type of SHAP explainer to use. Defaults to "tree".
    background_data : array-like, optional
        Background dataset required for the "kernel" explainer.
    n_background_samples : int, optional
        Number of background samples to subsample for the "kernel" explainer.
        Defaults to 100.

    Attributes
    ----------
    shap_values_ : np.ndarray of shape (n_samples, n_features)
        SHAP values computed after calling fit. For binary classification,
        values correspond to the positive class.
    feature_names_ : list of str
        Feature names inferred from the input data (DataFrame columns or
        auto-generated as f0, f1, ...).
    explainer_ : shap.Explainer
        Fitted SHAP explainer instance.

    Examples
    --------
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from caketool.explainability import ShapExplainer
    >>> model = GradientBoostingClassifier().fit(X_train, y_train)
    >>> explainer = ShapExplainer(model=model, explainer_type="tree")
    >>> explainer.fit(X_test)
    >>> importance = explainer.get_feature_importance()
    """

    def __init__(
        self,
        model,
        explainer_type: Literal["tree", "linear", "kernel"] = "tree",
        background_data=None,
        n_background_samples: int = 100,
    ) -> None:
        if explainer_type not in _VALID_EXPLAINER_TYPES:
            raise ValueError(f"Invalid explainer_type '{explainer_type}'. Must be one of {_VALID_EXPLAINER_TYPES}.")
        if explainer_type == "kernel" and background_data is None:
            raise ValueError("background_data is required for explainer_type='kernel'.")

        self.model = model
        self.explainer_type = explainer_type
        self.background_data = background_data
        self.n_background_samples = n_background_samples
        self._is_fitted = False

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(_NOT_FITTED_MSG)

    @require_dependencies("shap")
    def fit(self, X) -> ShapExplainer:
        """
        Compute SHAP values for the given input data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input data to explain.

        Returns
        -------
        self : ShapExplainer
        """
        import shap

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            self._X_fit = X.copy()
        else:
            self.feature_names_ = [f"f{i}" for i in range(X.shape[1])]
            self._X_fit = pd.DataFrame(X, columns=self.feature_names_)

        if self.explainer_type == "tree":
            self.explainer_ = shap.TreeExplainer(self.model)
        elif self.explainer_type == "linear":
            self.explainer_ = shap.LinearExplainer(self.model, X)
        else:  # kernel
            n = min(self.n_background_samples, len(self.background_data))
            bg = shap.sample(self.background_data, n)
            self.explainer_ = shap.KernelExplainer(self.model.predict_proba, bg)

        raw_values = self.explainer_.shap_values(X)

        # Normalise binary classification: take positive-class SHAP values
        if isinstance(raw_values, list):
            raw_values = raw_values[1] if len(raw_values) == 2 else raw_values[0]
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
            raw_values = raw_values[:, :, 1]

        self.shap_values_ = np.array(raw_values)
        self._is_fitted = True
        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return global feature importance based on mean absolute SHAP values.

        Returns
        -------
        pd.DataFrame
            Columns: feature, mean_abs_shap, rank. Sorted by mean_abs_shap descending,
            rank starts at 1.

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        self._check_fitted()
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        df = pd.DataFrame({"feature": self.feature_names_, "mean_abs_shap": mean_abs})
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df

    def get_local_explanation(self, X, row_index: int = 0) -> pd.DataFrame:
        """
        Return SHAP explanation for a single sample.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input data.
        row_index : int, optional
            Index of the sample to explain. Defaults to 0.

        Returns
        -------
        pd.DataFrame
            Columns: feature, feature_value, shap_value, abs_shap.
            Sorted by abs_shap descending.

        Raises
        ------
        IndexError
            If row_index is out of bounds.
        RuntimeError
            If called before fit.
        """
        self._check_fitted()
        n = len(X) if isinstance(X, pd.DataFrame) else X.shape[0]
        if row_index < 0 or row_index >= n:
            raise IndexError(f"row_index {row_index} is out of bounds for input with {n} rows.")

        feature_values = X.iloc[row_index].values if isinstance(X, pd.DataFrame) else X[row_index]
        shap_vals = self.shap_values_[row_index]

        df = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "feature_value": feature_values,
                "shap_value": shap_vals,
                "abs_shap": np.abs(shap_vals),
            }
        )
        return df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    def get_top_drivers(self, X, row_index: int = 0, n: int = 10) -> pd.DataFrame:
        """
        Return the top-N most influential features for a single sample.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input data.
        row_index : int, optional
            Index of the sample to explain. Defaults to 0.
        n : int, optional
            Number of top features to return. Defaults to 10.

        Returns
        -------
        pd.DataFrame
            Top-N rows from get_local_explanation with an added direction column
            ("positive" or "negative").

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        self._check_fitted()
        local_df = self.get_local_explanation(X, row_index=row_index)
        top_df = local_df.head(n).copy()
        top_df["direction"] = top_df["shap_value"].apply(lambda v: "positive" if v >= 0 else "negative")
        return top_df.reset_index(drop=True)

    def _get_base_value(self) -> float:
        """Extract base value from explainer for plotting."""
        expected = self.explainer_.expected_value
        if isinstance(expected, list | np.ndarray):
            arr = np.array(expected).flatten()
            return float(arr[1]) if len(arr) >= 2 else float(arr[0])
        return float(expected)

    @require_dependencies("shap")
    def show_summary(self, plot_type: str = "bar", max_display: int = 20) -> None:
        """
        Display a SHAP summary plot.

        Parameters
        ----------
        plot_type : {"bar", "beeswarm", "dot"}, optional
            Type of summary plot. Defaults to "bar".
        max_display : int, optional
            Maximum number of features to display. Defaults to 20.

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        self._check_fitted()
        import shap

        # Create Explanation object for newer SHAP API
        explanation = shap.Explanation(
            values=self.shap_values_,
            base_values=self._get_base_value(),
            data=self._X_fit.values if hasattr(self._X_fit, "values") else self._X_fit,
            feature_names=self.feature_names_,
        )

        if plot_type == "beeswarm":
            shap.plots.beeswarm(explanation, max_display=max_display)
        elif plot_type == "bar":
            shap.plots.bar(explanation, max_display=max_display)
        else:
            # Fallback for dot or other types
            shap.summary_plot(
                self.shap_values_,
                features=self._X_fit,
                feature_names=self.feature_names_,
                plot_type=plot_type,
                max_display=max_display,
            )

    @require_dependencies("shap")
    def show_waterfall(self, X, row_index: int = 0, max_display: int = 15) -> None:
        """
        Display a SHAP waterfall plot for a single sample.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
            Input data.
        row_index : int, optional
            Index of the sample to explain. Defaults to 0.
        max_display : int, optional
            Maximum number of features to display. Defaults to 15.

        Raises
        ------
        RuntimeError
            If called before fit.
        """
        self._check_fitted()
        import shap

        row_data = X.iloc[row_index].values if isinstance(X, pd.DataFrame) else X[row_index]

        explanation = shap.Explanation(
            values=self.shap_values_[row_index],
            base_values=self._get_base_value(),
            data=row_data,
            feature_names=self.feature_names_,
        )
        shap.waterfall_plot(explanation, max_display=max_display)

    @require_dependencies("shap")
    def show_dependence(self, feature: str, interaction_feature: str = "auto") -> None:
        """
        Display a SHAP dependence plot for a feature.

        Parameters
        ----------
        feature : str
            Feature name to plot.
        interaction_feature : str, optional
            Feature to use for colour encoding. Defaults to "auto".

        Raises
        ------
        ValueError
            If feature is not in feature_names_.
        RuntimeError
            If called before fit.
        """
        self._check_fitted()
        if feature not in self.feature_names_:
            raise ValueError(f"Feature '{feature}' not found in feature_names_. Available: {self.feature_names_}")

        import shap

        shap.dependence_plot(
            feature,
            self.shap_values_,
            self._X_fit,
            interaction_index=interaction_feature,
        )
