import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

class AdversarialModel:
    """
    A class designed to detect drift between two datasets by training a binary
    classifier to distinguish between them. Drift is detected when the model can
    successfully differentiate between the two datasets, indicating that they
    come from different distributions.

    Attributes:
    -----------
    model : object, optional
        A machine learning classifier used to detect drift (default is CatBoostClassifier).
    auc_score : float
        The ROC AUC score representing the ability of the model to distinguish 
        between the two datasets (high score indicates drift).

    Methods:
    --------
    fit(df1, df2, groups_col=["label"], features=None):
        Fits the adversarial classifier to the concatenated data to detect drift.
    show(n_features=5):
        Displays the ROC AUC score and the top N most important features contributing to drift.
    """
    
    def __init__(self, model=None) -> None:
        """
        Initializes the AdversarialModel with a given model or a default 
        CatBoostClassifier if none is provided.

        Parameters:
        -----------
        model : object, optional
            A machine learning classifier (default is CatBoostClassifier).
        """
        self.model = model or CatBoostClassifier(verbose=False)
        self.auc_score = -1

    def fit(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        groups_col=["label"],
        features=None,
    ):
        """
        Fits the adversarial classifier to detect drift between two datasets by
        differentiating between them. If the model can classify data points 
        from df1 and df2 effectively, this indicates that drift has occurred.

        Parameters:
        -----------
        df1 : pd.DataFrame
            The first dataset representing the reference distribution (e.g., training data).
        df2 : pd.DataFrame
            The second dataset to compare against the reference (e.g., new or testing data).
        groups_col : list, optional
            Column(s) used for stratification during the train-test split (default is ["label"]).
        features : list, optional
            List of feature names to be used for training (default is the intersection of the columns in both datasets).
        
        Returns:
        --------
        None
        """
        # Concatenating the two datasets and assigning labels (0 for df1, 1 for df2)
        data_adversarial = pd.concat(
            [
                df1.assign(label=0),
                df2.assign(label=1),
            ],
            ignore_index=True,
        )
        
        # If features are not provided, use the common columns between the two datasets
        if features == None:
            features = list(set(df1.columns).intersection(df2.columns))
        
        # Splitting the data into training and validation sets using stratified sampling
        X_train, X_val, y_train, y_val = train_test_split(
            data_adversarial[features], 
            data_adversarial["label"], 
            test_size=0.2, 
            stratify=data_adversarial[groups_col].apply(lambda x: "_".join(map(str, x)), axis=1)
        )
        
        # Fitting the model
        self.model.fit(X_train, y_train)
        
        # Calculating the ROC AUC score to measure the ability to detect drift
        self.auc_score = roc_auc_score(y_val, self.model.predict(X_val))

    def show(self, n_features=5):
        """
        Displays the ROC AUC score of the model, which represents the ability
        to detect drift between the two datasets. It also shows the top N 
        important features contributing to drift detection.

        Parameters:
        -----------
        n_features : int, optional
            The number of top important features to display (default is 5).

        Returns:
        --------
        None
        """
        print(f"ROC AUC: {self.auc_score:02f}")
        print(f"Top {n_features} important feature(s) contributing to drift:")
        print(f"{self.model.get_feature_importance(prettified=True).head(n_features)}")