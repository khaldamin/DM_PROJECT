# src/models/svm_model.py
"""Support Vector Machine wrapper for the Cupid project.
Implements the same BaseModel interface as the other classifiers.
"""

from sklearn.svm import SVC
from .base_model import BaseModel


class CupidSVM(BaseModel):
    """SVM classifier for Cupid Matchmaking Compatibility.

    Parameters are chosen to be reproducible and comparable with the other
    models. ``probability=True`` enables ``predict_proba`` which is required
    for ROC‑AUC and SHAP analysis.
    """

    def __init__(self,
                 C: float = 1.0,
                 kernel: str = "rbf",
                 gamma: str = "scale",
                 probability: bool = True,
                 random_state: int = 42,
                 **kwargs):
        super().__init__("SVM")
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.kwargs = kwargs

        svm_params = {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "probability": self.probability,
            "random_state": self.random_state,
        }
        svm_params.update(self.kwargs)
        self.model = SVC(**svm_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

