import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Gini coefficient for a binary classifier.

    Defined as ``GINI = 2 * ROC_AUC - 1``, which maps ROC-AUC from [0.5, 1.0]
    to a Gini range of [0, 1].  A Gini of 0 corresponds to a random model
    (AUC = 0.5) and 1 to a perfect model (AUC = 1.0).

    Parameters
    ----------
    y_actual : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or decision scores for the positive class.

    Returns
    -------
    float
        Gini coefficient in the range [0, 1].

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> X, y = make_classification(n_samples=200, random_state=0)
    >>> model = LogisticRegression().fit(X, y)
    >>> gini(y, model.predict_proba(X)[:, 1])
    0.97...
    """
    return 2 * roc_auc_score(y_actual, y_pred) - 1
