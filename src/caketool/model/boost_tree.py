import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import set_config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .feature_encoder import FeatureEncoder
from .feature_remover import ColinearFeatureRemover, UnivariateFeatureRemover
from .infinity_handler import InfinityHandler

set_config(transform_output="pandas")


DEFAULT_PARAM = {
    "feature_encoder": {
        "encoder_name": "category_encoders.TargetEncoder",
    },
    "colinear_feature_remover": {"correlation_threshold": 0.9},
    "univariate_feature_remover": {
        "score_func": f_classif,
        "threshold": 0.05,
    },
    "model_params": {
        "random_state": 8799,
        "booster": "gbtree",
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "grow_policy": "lossguide",
        "max_depth": 7,
        "eta": 0.05,
        "gamma": 0.5,
        "subsample": 0.65,
        "min_child_weight": 16,
        "colsample_bytree": 0.5,
        "scale_pos_weight": 1,
        "nthread": 4,
    },
}


class BoostTree(BaseEstimator, RegressorMixin):
    """
    XGBoost-based binary classifier with a built-in preprocessing pipeline.

    The pipeline applies, in order:
    1. Categorical encoding (default: TargetEncoder)
    2. Infinity value handling
    3. Univariate feature removal (by p-value threshold)
    4. Collinear feature removal (by correlation threshold)

    Parameters
    ----------
    param : dict, optional
        Configuration dict with keys:
        - ``feature_encoder``: kwargs for FeatureEncoder
        - ``univariate_feature_remover``: kwargs for UnivariateFeatureRemover
        - ``colinear_feature_remover``: kwargs for ColinearFeatureRemover
        - ``model_params``: kwargs for xgb.XGBClassifier
        Defaults to DEFAULT_PARAM.

    Examples
    --------
    >>> model = BoostTree()
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> proba = model.predict_proba(X_test)[:, 1]
    """

    def __init__(self, param: dict = DEFAULT_PARAM) -> None:
        super().__init__()
        self.param = param
        feature_encoder = FeatureEncoder(**param["feature_encoder"])
        infinity_handler = InfinityHandler()
        univariate_feature_remover = UnivariateFeatureRemover(**param["univariate_feature_remover"])
        colinear_feature_remover = ColinearFeatureRemover(**param["colinear_feature_remover"])
        self.model = xgb.XGBClassifier(**param["model_params"])
        self.preprocess = Pipeline(
            [
                ("feature_encoder", feature_encoder),
                ("infinity_handler", infinity_handler),
                ("univariate_feature_remover", univariate_feature_remover),
                ("colinear_feature_remover", colinear_feature_remover),
            ]
        )
        self.pipeline = Pipeline(
            [
                ("preprocess", self.preprocess),
                ("model", self.model),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set: list | None = None, verbose: bool = False) -> "BoostTree":
        """
        Fit the preprocessing pipeline and XGBoost model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Binary target labels (0 or 1).
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping. Each element is preprocessed
            automatically before being passed to XGBoost.
        verbose : bool, optional
            Whether to print XGBoost training logs. Defaults to False.

        Returns
        -------
        self : BoostTree
        """
        self.preprocess.fit(X, y)
        if eval_set is not None and len(eval_set) > 0:
            eval_set = [(self.preprocess.transform(s[0]), s[1]) for s in eval_set]
        self.pipeline.fit(
            X,
            y,
            model__eval_set=eval_set,
            model__verbose=verbose,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns are [P(class=0), P(class=1)].
        """
        return self.pipeline.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances across all score types.

        Returns
        -------
        pd.DataFrame
            Columns: feature_name, gain, cover, total_gain, total_cover, weight.
        """
        feat_importance = None
        for score_type in ["gain", "cover", "total_gain", "total_cover", "weight"]:
            fi_dict = self.model.get_booster().get_score(importance_type=score_type)
            fi_tb = pd.DataFrame(list(fi_dict.items()), columns=["feature_name", score_type])

            if feat_importance is not None:
                feat_importance = feat_importance.merge(fi_tb, on="feature_name")
            else:
                feat_importance = fi_tb

        return feat_importance

    def get_feature_names(self) -> list[str]:
        """Return the list of feature names used by the booster."""
        return self.model.get_booster().feature_names

    def fit_oof(
        X: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series | None = None,
        params: dict = DEFAULT_PARAM,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: int = 42,
    ) -> tuple[list["BoostTree"], np.ndarray, np.ndarray]:
        """
        Fit models using out-of-fold (OOF) cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Binary target labels.
        groups : pd.Series, optional
            Group labels for stratified splitting. Defaults to y.
        params : dict, optional
            BoostTree parameter dict. Defaults to DEFAULT_PARAM.
        n_splits : int, optional
            Number of CV folds. Defaults to 5.
        n_repeats : int, optional
            Number of times to repeat CV. Defaults to 1.
        random_state : int, optional
            Random seed. Defaults to 42.

        Returns
        -------
        models : list[BoostTree]
            Fitted model for each fold.
        oof_predictions : np.ndarray
            Concatenated OOF probability predictions.
        oof_labels : np.ndarray
            Concatenated OOF true labels.
        """
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        oof_predictions = []
        oof_labels = []
        models = []

        for _, (train_idx, val_idx) in tqdm(enumerate(skf.split(X, groups))):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model = BoostTree(params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            val_pred = model.predict_proba(X_val)
            models.append(model)
            oof_predictions.append(val_pred)
            oof_labels.append(y_val)

        return models, np.concatenate(oof_predictions), np.concatenate(oof_labels)


class EnsembleBoostTree(BaseEstimator, RegressorMixin):
    """
    Ensemble of BoostTree models that averages predictions across all estimators.

    Each estimator may have been trained on a different feature subset or fold.
    Predictions are automatically restricted to each estimator's own feature set.

    Parameters
    ----------
    estimators : list[BoostTree]
        Fitted BoostTree models to ensemble.

    Examples
    --------
    >>> models, _, _ = BoostTree.fit_oof(X_train, y_train)
    >>> ensemble = EnsembleBoostTree(models)
    >>> proba = ensemble.predict_proba(X_test)[:, 1]
    """

    def __init__(self, estimators: list[BoostTree]) -> None:
        self.estimators = estimators

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict by averaging class predictions from all estimators.

        Parameters
        ----------
        X : pd.DataFrame
            Input features (must contain all feature columns).

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        y_preds = [estimator.predict(X[estimator.get_feature_names()]) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities by averaging across all estimators.

        Parameters
        ----------
        X : pd.DataFrame
            Input features (must contain all feature columns).

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        y_preds = [estimator.predict_proba(X[estimator.get_feature_names()]) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importance: pd.DataFrame = None
        for estimator in self.estimators:
            sub_fi: pd.DataFrame = estimator.get_feature_importance()
            sub_fi["num_tree"] = 1
            if feature_importance is None:
                feature_importance: pd.DataFrame = sub_fi
            else:
                feature_importance = feature_importance.merge(sub_fi, how="outer", on="feature_name").fillna(0)
                for score_type in sub_fi.columns[1:]:
                    feature_importance[score_type] = (
                        feature_importance[score_type + "_x"] + feature_importance[score_type + "_y"]
                    )
                feature_importance = feature_importance[sub_fi.columns]
        return feature_importance

    def get_feature_names(self) -> list[str]:
        feature_names = set()
        for estimator in self.estimators:
            feature_names.update(estimator.get_feature_names())
        return sorted(list(feature_names))
