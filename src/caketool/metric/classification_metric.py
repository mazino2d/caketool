from sklearn.metrics import roc_auc_score


def gini(y_oot, y_pred):
    return 2 * roc_auc_score(y_oot, y_pred) - 1
