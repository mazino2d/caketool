import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from caketool.model.classification.binary import BinaryBoostTree
from caketool.model.config import ModelConfig

# Disable univariate filtering by default: adversarial labels are random when
# distributions match, so all features would be dropped with the standard 0.05 threshold.
_ADVERSARIAL_DEFAULT_CONFIG = ModelConfig(univariate_threshold=1.0)

# Synthetic target column: 0 for ``df1`` rows, 1 for ``df2`` rows
_LABEL_COL = "label"


class AdversarialModel:
    """Detect distribution drift between two datasets via adversarial classification.

    Trains a ``BinaryBoostTree`` to distinguish samples from ``df1`` (label=0)
    vs ``df2`` (label=1). A high AUC indicates the model can tell them apart —
    i.e. the distributions have drifted.

    Categorical features are handled automatically by the preprocessing pipeline.

    Parameters
    ----------
    config : ModelConfig or None
        Configuration for the underlying ``BinaryBoostTree``. Uses
        ``ModelConfig()`` defaults when ``None``.
    random_state : int or None
        Seed for the train/validation split. Pass an int for reproducible
        results. Defaults to ``None``.

    Attributes
    ----------
    auc_score : float
        ROC AUC on the held-out validation split. ``-1`` before ``fit``.
    model_ : BinaryBoostTree
        The fitted adversarial classifier (available after ``fit``).

    Examples
    --------
    >>> adv = AdversarialModel()
    >>> adv.fit(df_train, df_new)
    >>> drift_df = adv.get_drift_features()
    >>> print(f"AUC: {adv.auc_score:.4f}")
    """

    def __init__(self, config: ModelConfig | None = None, random_state: int | None = None) -> None:
        self.config = config if config is not None else _ADVERSARIAL_DEFAULT_CONFIG
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
            data[self.feature_names_],
            data[_LABEL_COL],
            test_size=0.2,
            stratify=data[groups_col].apply(lambda x: "_".join(map(str, x)), axis=1),
            random_state=self.random_state,
        )

        self.model_ = BinaryBoostTree(self.config)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        self.auc_score = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
        return self

    def get_drift_features(self) -> pd.DataFrame:
        """Return all features ranked by their contribution to drift.

        Features with high ``gain_pct`` are the strongest signals of
        distribution shift between the two datasets.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature_name``, ``gain``, ``gain_pct``, sorted by
            ``gain_pct`` descending.
        """
        fi = self.model_.get_feature_importance()
        return fi[["feature_name", "gain", "gain_pct"]].sort_values("gain_pct", ascending=False).reset_index(drop=True)
