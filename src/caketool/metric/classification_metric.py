import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate gini score for two arrays. GINI = 2* ROC_AUC - 1

    Parameters
    ----------
        y_actual : list-like
            Array of actual values
        y_pred : list-like
            Array of predicted values

    Returns
    -------
        A single float number
    """
    return 2 * roc_auc_score(y_actual, y_pred) - 1
