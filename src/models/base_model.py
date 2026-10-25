import os
import joblib
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all Cupid classification models.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support predict_proba.")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)
