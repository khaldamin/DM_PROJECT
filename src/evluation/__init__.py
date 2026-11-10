# Deprecated legacy alias: redirected to src.evaluation
from src.evaluation.metrics import evaluate_classification_model, print_evaluation_summary
from src.evaluation.cross_validation import run_cross_validation
from src.evaluation.tuning import tune_hyperparameters

__all__ = [
    "evaluate_classification_model",
    "print_evaluation_summary",
    "run_cross_validation",
    "tune_hyperparameters"
]
