from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

class CupidDecisionTree(BaseModel):
    """
    Decision Tree Classifier for Cupid Matchmaking Compatibility.
    """
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini', random_state=42):
        super().__init__("Decision Tree")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.feature_importances_
