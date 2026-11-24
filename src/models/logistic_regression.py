from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class CupidLogisticRegression(BaseModel):
    """
    Logistic Regression Classifier for Cupid Matchmaking Compatibility.
    """
    def __init__(self, C=1.0, penalty=None, solver='lbfgs', max_iter=1000, random_state=42, **kwargs):
        super().__init__("Logistic Regression")
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs

        lr_params = {
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "random_state": self.random_state
        }
        # Only pass penalty if explicitly specified and not default l2 to avoid scikit-learn >= 1.8 FutureWarning
        if self.penalty is not None and self.penalty != 'l2':
            lr_params["penalty"] = self.penalty
        lr_params.update(kwargs)

        self.model = LogisticRegression(**lr_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
