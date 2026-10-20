"""Model building utilities: feature selection and out-of-fold training.

Works with any scikit-learn compatible estimator instead of wrapping a
specific model library.

Feature selection
-----------------
- ``UnivariateFeatureSelector`` – keeps features whose univariate test
  (F-test by default) against the target is significant.
- ``ColinearFeatureSelector`` – drops features highly correlated with a more
  target-relevant feature.

Out-of-fold training
--------------------
- ``fit_oof`` – trains one clone of an estimator per CV fold and returns
  unbiased out-of-fold predictions (``OOFResult``).
- ``OOFEnsemble`` – averages the fold estimators' predictions on new data.

Example
-------
>>> from sklearn.pipeline import make_pipeline
>>> from xgboost import XGBClassifier
>>> from caketool.model import ColinearFeatureSelector, UnivariateFeatureSelector, fit_oof
>>> pipeline = make_pipeline(UnivariateFeatureSelector(), ColinearFeatureSelector(), XGBClassifier())
>>> result = fit_oof(pipeline, X_train, y_train, cv=5)
>>> proba = result.ensemble.predict_proba(X_test)[:, 1]
"""

from .oof import OOFEnsemble as OOFEnsemble
from .oof import OOFResult as OOFResult
from .oof import fit_oof as fit_oof
from .selection import ColinearFeatureSelector as ColinearFeatureSelector
from .selection import UnivariateFeatureSelector as UnivariateFeatureSelector
