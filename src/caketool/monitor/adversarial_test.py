import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from caketool.model.classification.binary import BinaryBoostTree
from caketool.model.config import ModelConfig

# Disable univariate filtering by default: adversarial labels are random when
# distributions match, so all features would be dropped with the standard 0.05 threshold.
_ADVERSARIAL_DEFAULT_CONFIG = ModelConfig(univariate_threshold=1.0)


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

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config if config is not None else _ADVERSARIAL_DEFAULT_CONFIG
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
            Columns used for stratified splitting. Default ``["label"]``.
        features : list[str], optional
            Feature columns to use. Defaults to the intersection of both
            DataFrames' columns.

        Returns
        -------
        self
        """
        if groups_col is None:
            groups_col = ["label"]

        data = pd.concat([df1.assign(label=0), df2.assign(label=1)], ignore_index=True)

        if features is None:
            features = list(set(df1.columns).intersection(df2.columns))
        self.feature_names_ = features

        X_train, X_val, y_train, y_val = train_test_split(
            data[features],
            data["label"],
            test_size=0.2,
            stratify=data[groups_col].apply(lambda x: "_".join(map(str, x)), axis=1),
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
