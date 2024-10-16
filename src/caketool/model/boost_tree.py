from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.feature_selection import f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

set_config(transform_output = "pandas")
from caketool.feature.feature_encoder import FeatureEncoder
from caketool.feature.feature_remover import ColinearFeatureRemover, UnivariateFeatureRemover
from caketool.feature.infinity_handler import InfinityHandler
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb 

DEFAULT_PARAM = {
    'feature_encoder': {
        "encoder_name": "category_encoders.TargetEncoder",
    },
    'colinear_feature_remover': {
        "correlation_threshold": 0.9
    },
    'univariate_feature_remover': {
        "score_func": f_classif,
        "threshold": 0.05,
    },
    'model_params': {
        'random_state': 8799,
        'booster': 'gbtree',
        'tree_method': 'approx',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'grow_policy': 'lossguide',
        'max_depth': 7,
        'eta': 0.05,
        'gamma': 0.5,
        'subsample': 0.65,
        'min_child_weight': 16,
        'colsample_bytree': 0.5, 
        'scale_pos_weight': 1, 
        'nthread': 4,
    }
}

class BoostTree(BaseEstimator, RegressorMixin):
  
    def __init__(self, param: Dict = DEFAULT_PARAM):
        super().__init__()
        self.param = param
        feature_encoder = FeatureEncoder(**param["feature_encoder"])
        infinity_handler = InfinityHandler()
        univariate_feature_remover = UnivariateFeatureRemover(**param["univariate_feature_remover"])
        colinear_feature_remover = ColinearFeatureRemover(**param["colinear_feature_remover"])
        self.model = xgb.XGBClassifier(**param["model_params"])
        self.preprocess = Pipeline([
            ('feature_encoder', feature_encoder),
            ('infinity_handler', infinity_handler),
            ('univariate_feature_remover', univariate_feature_remover),
            ('colinear_feature_remover', colinear_feature_remover)
        ])
        self.pipeline = Pipeline([
            ('preprocess', self.preprocess),
            ('model', self.model),
        ])

    def fit(self, X, y, eval_set=None, verbose=False):
        self.preprocess.fit(X, y)
        if eval_set is not None and len(eval_set) > 0:
            eval_set = [(self.preprocess.transform(s[0]), s[1]) for s in eval_set]
        self.pipeline.fit(
            X, y,
            model__eval_set = eval_set,
            model__verbose = verbose,
        )
        return self

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.pipeline.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        feat_importance = None
        for score_type in ["gain", "cover", "total_gain", "total_cover", "weight"]:
            fi_dict = self.model.get_booster().get_score(importance_type=score_type)
            fi_tb = pd.DataFrame(list(fi_dict.items()), columns=["feature_name", score_type])

            if feat_importance is not None:
                feat_importance = feat_importance.merge(fi_tb, on="feature_name")
            else:
                feat_importance = fi_tb

        return feat_importance
    
    def get_feature_names(self) -> List[str]:
        return self.model.get_booster().feature_names

    def fit_oof(X, y, groups=None, params=DEFAULT_PARAM, n_splits=5, n_repeats=1, random_state=42):
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
  
    def __init__(self, estimators: List[BoostTree]):
        self.estimators = estimators

    def predict(self, X: pd.DataFrame):
        y_preds = [estimator.predict(X[estimator.get_feature_names()]) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X: pd.DataFrame):
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
                feature_importance = feature_importance.merge(
                    sub_fi,
                    how="outer", on="feature_name"
                ).fillna(0)
                for score_type in sub_fi.columns[1:]:
                    feature_importance[score_type] = feature_importance[score_type + "_x"] + feature_importance[score_type + "_y"]
                feature_importance = feature_importance[sub_fi.columns]
        return feature_importance
    
    def get_feature_names(self) -> List[str]:
        feature_names = set()
        for estimator in self.estimators:
            feature_names.update(estimator.get_feature_names())
        return sorted(list(feature_names))