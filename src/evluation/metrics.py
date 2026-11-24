# Backward compatibility alias - forward to src.evaluation.metrics
from src.evaluation.metrics import evaluate_classification_model, print_evaluation_summary

__all__ = ["evaluate_classification_model", "print_evaluation_summary"]
