from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class CupidLogisticRegression(BaseModel):
    """
    Logistic Regression Classifier for Cupid Matchmaking Compatibility.
    """
    def __init__(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42):
        super().__init__("Logistic Regression")
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
