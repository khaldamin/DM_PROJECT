import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.data.preprocessing import CupidDataPreprocessor
from .metrics import evaluate_classification_model

def run_cross_validation(model_class, model_params, X, y=None, n_splits=5, random_state=42,
                         raw_data=False, target_col='compatible'):
    """
    Runs Stratified K-Fold Cross Validation for a given model class.

    When ``raw_data`` is True, X must be a raw DataFrame containing the target
    column. A fresh preprocessor is fitted on each fold's training rows only.
    The default keeps the original already-preprocessed array API intact.
    """
    if raw_data:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame when raw_data=True.")
        if target_col not in X.columns:
            raise ValueError(f"Target column '{target_col}' is missing from X.")
        y = X[target_col].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_metrics = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC-AUC': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if raw_data:
            train_df = X.iloc[train_idx]
            validation_df = X.iloc[val_idx]
            preprocessor = CupidDataPreprocessor(target_col=target_col)
            X_tr, y_tr = preprocessor.fit_transform(train_df)
            X_va = preprocessor.transform(validation_df)
            y_va = validation_df[target_col].to_numpy()
        else:
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
