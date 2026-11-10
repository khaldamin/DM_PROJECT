import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from src.evaluation.metrics import evaluate_classification_model, print_evaluation_summary
from src.evaluation.cross_validation import run_cross_validation
from src.evaluation.tuning import tune_hyperparameters
from src.models.logistic_regression import CupidLogisticRegression

@pytest.fixture
def eval_data():
    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = np.random.binomial(1, 0.5, 80)
    return X, y

def test_evaluate_classification_model_keys(eval_data):
    X, y = eval_data
    model = CupidLogisticRegression(max_iter=100, random_state=42)
    model.fit(X, y)
    metrics = evaluate_classification_model(model, X, y, model_name="TestLogReg")
    for key in ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "LogLoss", "Confusion_Matrix", "Classification_Report"]:
        assert key in metrics

def test_evaluate_classification_model_confusion_matrix(eval_data):
    X, y = eval_data
    model = CupidLogisticRegression(max_iter=100, random_state=42)
    model.fit(X, y)
    metrics = evaluate_classification_model(model, X, y)
    cm = np.array(metrics["Confusion_Matrix"])
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y)

def test_evaluate_classification_model_log_loss(eval_data):
    X, y = eval_data
    model = CupidLogisticRegression(max_iter=100, random_state=42)
    model.fit(X, y)
    metrics = evaluate_classification_model(model, X, y)
    assert not np.isnan(metrics["LogLoss"])
    assert metrics["LogLoss"] > 0

def test_print_evaluation_summary_runs_without_error(eval_data, capsys):
    X, y = eval_data
    model = CupidLogisticRegression(max_iter=100, random_state=42)
    model.fit(X, y)
    metrics = evaluate_classification_model(model, X, y)
    print_evaluation_summary(metrics)
    captured = capsys.readouterr()
    assert "Evaluation Metrics for" in captured.out
    assert "Accuracy" in captured.out

def test_run_cross_validation_splits(eval_data):
    X, y = eval_data
    summary, cv_metrics = run_cross_validation(
        CupidLogisticRegression, {"max_iter": 100, "random_state": 42}, X, y, n_splits=3
    )
    assert len(cv_metrics["Accuracy"]) == 3
    assert len(cv_metrics["F1-Score"]) == 3

def test_run_cross_validation_summary_metrics(eval_data):
    X, y = eval_data
    summary, _ = run_cross_validation(
        CupidLogisticRegression, {"max_iter": 100, "random_state": 42}, X, y, n_splits=3
    )
    assert "mean_Accuracy" in summary
    assert "mean_F1-Score" in summary
    assert 0.0 <= summary["mean_Accuracy"] <= 1.0

def test_tune_hyperparameters_grid(eval_data):
    X, y = eval_data
    base = LogisticRegression(random_state=42)
    param_grid = {"C": [0.1, 1.0]}
    best_est, best_params, best_score = tune_hyperparameters(
        base, param_grid, X, y, method="grid", cv=2, scoring="accuracy"
    )
    assert "C" in best_params
    assert best_score > 0.0

def test_tune_hyperparameters_random(eval_data):
    X, y = eval_data
    base = LogisticRegression(random_state=42)
    param_dist = {"C": [0.01, 0.1, 1.0, 10.0]}
    best_est, best_params, best_score = tune_hyperparameters(
        base, param_dist, X, y, method="random", n_iter=2, cv=2, scoring="accuracy"
    )
    assert "C" in best_params
    assert best_score > 0.0

