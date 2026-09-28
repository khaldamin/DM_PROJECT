from xgboost import XGBClassifier
from .base_model import BaseModel

class CupidXGBoost(BaseModel):
    """
    XGBoost Classifier for Cupid Matchmaking Compatibility.
    """
    def __init__(self, n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42):
        super().__init__("XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            eval_metric='logloss'
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def get_feature_importances(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.feature_importances_
