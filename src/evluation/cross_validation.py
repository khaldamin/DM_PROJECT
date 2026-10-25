import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .metrics import evaluate_classification_model

def run_cross_validation(model_class, model_params, X, y, n_splits=5, random_state=42):
    """
    Runs Stratified K-Fold Cross Validation for a given model class.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC-AUC': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_tr, y_tr)
        
        metrics = evaluate_classification_model(model, X_va, y_va)
        for key in cv_metrics.keys():
            cv_metrics[key].append(metrics[key])

    summary = {}
    for key, values in cv_metrics.items():
        summary[f"mean_{key}"] = np.mean(values)
        summary[f"std_{key}"] = np.std(values)

    return summary, cv_metrics
