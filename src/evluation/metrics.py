import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)

def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    """
    Computes standard classification evaluation metrics for a trained model.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
    loss = log_loss(y_test, y_proba) if y_proba is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "LogLoss": loss,
        "Confusion_Matrix": cm.tolist(),
        "Classification_Report": classification_report(y_test, y_pred, output_dict=True)
    }

    return metrics

def print_evaluation_summary(metrics: dict):
    """
    Prints formatted evaluation metrics.
    """
    print(f"\n==========================================")
    print(f" Evaluation Metrics for: {metrics['Model']}")
    print(f"==========================================")
    print(f" Accuracy    : {metrics['Accuracy']:.4f}")
    print(f" Precision   : {metrics['Precision']:.4f}")
    print(f" Recall      : {metrics['Recall']:.4f}")
    print(f" F1-Score    : {metrics['F1-Score']:.4f}")
    if not np.isnan(metrics['ROC-AUC']):
        print(f" ROC-AUC     : {metrics['ROC-AUC']:.4f}")
    if not np.isnan(metrics['LogLoss']):
        print(f" Log Loss    : {metrics['LogLoss']:.4f}")
    print(" Confusion Matrix:")
    print(np.array(metrics['Confusion_Matrix']))
    print("==========================================\n")
