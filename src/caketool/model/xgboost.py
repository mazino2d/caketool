import xgboost as xgb
from sklearn.model_selection import train_test_split


class XCakeGboost(xgb.XGBClassifier):
    def fit(self, X, y, val_size=0.2, stratify=None, verbose=False, **kwargs) -> "XCakeGboost":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=val_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=stratify
        )
        return super().fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], verbose=verbose, **kwargs)
