from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class CupidRandomForest(BaseModel):
    """
    Random Forest Classifier for Cupid Matchmaking Compatibility.
    """
    def __init__(self, n_estimators=100, max_depth=12, min_samples_split=5, min_samples_leaf=2, n_jobs=-1, random_state=42):
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.feature_importances_
