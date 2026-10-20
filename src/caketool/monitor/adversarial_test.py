import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

_DEFAULT_PARAMS = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1}

# Synthetic target column: 0 for ``df1`` rows, 1 for ``df2`` rows
_LABEL_COL = "label"


class AdversarialModel:
    """Detect distribution drift between two datasets via adversarial classification.

    Trains an ``XGBClassifier`` to distinguish samples from ``df1`` (label=0)
    vs ``df2`` (label=1). A high AUC indicates the model can tell them apart —
    i.e. the distributions have drifted.

    String, object and categorical columns are passed to XGBoost as native
    categoricals; infinities are treated as missing values.

    Parameters
    ----------
    params : dict or None
        Keyword arguments for ``XGBClassifier``, merged over the defaults
        ``n_estimators=200, max_depth=4, learning_rate=0.1``.
    random_state : int or None
        Seed for the train/validation split and the classifier. Pass an int
        for reproducible results. Defaults to ``None``.

    Attributes
    ----------
    auc_score : float
        ROC AUC on the held-out validation split. ``-1`` before ``fit``.
    model_ : xgboost.XGBClassifier
        The fitted adversarial classifier (available after ``fit``).

    Examples
    --------
    >>> adv = AdversarialModel()
    >>> adv.fit(df_train, df_new)
    >>> drift_df = adv.get_drift_features()
    >>> print(f"AUC: {adv.auc_score:.4f}")
    """

    def __init__(self, params: dict | None = None, random_state: int | None = None) -> None:
        self.params = params
        self.random_state = random_state
        self.auc_score = -1

    def fit(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        groups_col: list[str] | None = None,
        features: list[str] | None = None,
    ) -> "AdversarialModel":
        """Fit the adversarial classifier to detect drift.

        Parameters
        ----------
        df1 : pd.DataFrame
            Reference dataset (e.g. training data). Assigned label ``0``.
        df2 : pd.DataFrame
            Comparison dataset (e.g. new production data). Assigned label ``1``.
        groups_col : list[str], optional
            Extra columns used for stratified splitting, on top of the
            synthetic ``label`` (which is always used). Default ``None``.
        features : list[str], optional
            Feature columns to use. Defaults to the sorted intersection of both
            DataFrames' columns, excluding ``label`` and ``groups_col``.

        Raises
        ------
        ValueError
            If ``features`` contains ``label``, which is the synthetic target.

        Notes
        -----
        A ``label`` column already present in ``df1`` or ``df2`` is
        overwritten by the synthetic target and never used as a feature.

        Returns
        -------
        self
        """
        groups_col = [_LABEL_COL] + [c for c in (groups_col or []) if c != _LABEL_COL]

        if _LABEL_COL in df1.columns or _LABEL_COL in df2.columns:
            warnings.warn(
                f"Input column {_LABEL_COL!r} is overwritten by the adversarial target and not used as a feature.",
                UserWarning,
                stacklevel=2,
            )

        data = pd.concat([df1.assign(**{_LABEL_COL: 0}), df2.assign(**{_LABEL_COL: 1})], ignore_index=True)

        if features is None:
            features = sorted(set(df1.columns).intersection(df2.columns) - set(groups_col))
        elif _LABEL_COL in features:
            raise ValueError(f"{_LABEL_COL!r} is the adversarial target and cannot be used as a feature.")
        self.feature_names_ = list(features)

        X_train, X_val, y_train, y_val = train_test_split(
            _prepare_features(data[self.feature_names_]),
            data[_LABEL_COL],
            test_size=0.2,
            stratify=data[groups_col].apply(lambda x: "_".join(map(str, x)), axis=1),
            random_state=self.random_state,
        )

        params = {**_DEFAULT_PARAMS, "random_state": self.random_state, **(self.params or {})}
        self.model_ = XGBClassifier(**params, enable_categorical=True, tree_method="hist")
        self.model_.fit(X_train, y_train)

        self.auc_score = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
        return self

    def get_drift_features(self) -> pd.DataFrame:
        """Return all features ranked by their contribution to drift.

        Features with high ``gain_pct`` are the strongest signals of
        distribution shift between the two datasets.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature_name``, ``gain`` (total gain), ``gain_pct``,
            sorted by ``gain_pct`` descending. Unused features have zero gain.
        """
        gain = self.model_.get_booster().get_score(importance_type="total_gain")
        fi = pd.DataFrame(
            {"feature_name": self.feature_names_, "gain": [float(gain.get(f, 0.0)) for f in self.feature_names_]}
        )
        total = fi["gain"].sum()
        fi["gain_pct"] = fi["gain"] / total if total > 0 else 1.0 / len(fi)
        return fi.sort_values("gain_pct", ascending=False, kind="stable").reset_index(drop=True)


def _prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cast text columns to ``category``, bools to float and infinities to NaN for XGBoost."""
    X = X.copy()
    for col in X.columns:
        dtype = X[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_bool_dtype(dtype):
            X[col] = X[col].astype("float")
        elif pd.api.types.is_numeric_dtype(dtype):
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        else:
            X[col] = X[col].astype("string").astype("category")
    return X
