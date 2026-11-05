import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath('.'))

import joblib
from src.data.preprocessing import CupidDataPreprocessor

from src.models import (
    CupidLogisticRegression,
    CupidDecisionTree,
    CupidRandomForest,
    CupidXGBoost
)
from src.evluation.metrics import evaluate_classification_model, print_evaluation_summary
from src.evluation.tuning import tune_hyperparameters
from src.evluation.cross_validation import run_cross_validation


class CupidFoldPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn adapter that fits Cupid preprocessing on each CV fold."""

    def __init__(self, target_col='compatible'):
        self.target_col = target_col

    def fit(self, X, y=None):
        self.preprocessor_ = CupidDataPreprocessor(target_col=self.target_col)
        train_df = X.copy()
        if y is not None:
            train_df[self.target_col] = np.asarray(y)
        self.preprocessor_.fit_transform(train_df)
        self.feature_names_ = self.preprocessor_.feature_names
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X)

def main():
    print("==================================================")
    print(" Cupid Matchmaking Model Training & Evaluation")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    results_dir = "results"
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Preprocessing & Feature Engineering
    print("\n[1/5] Loading data and performing Feature Engineering...")
    raw_df = pd.read_csv(data_path)
    raw_train_df, raw_test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['compatible']
    )
    y_train = raw_train_df['compatible'].to_numpy()
    y_test = raw_test_df['compatible'].to_numpy()

    # Fit all final preprocessing statistics on precisely the outer training rows.
    preprocessor = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_train, _ = preprocessor.fit_transform(raw_train_df)
    X_test = preprocessor.transform(raw_test_df)
    print(f"Preprocessed train shape: {X_train.shape}, test shape: {X_test.shape}")

    # 2. Model Dictionary
    models = {
        "Logistic Regression": CupidLogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": CupidDecisionTree(max_depth=12, min_samples_split=10, random_state=42),
        "Random Forest": CupidRandomForest(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
        "XGBoost": CupidXGBoost(n_estimators=150, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=42)
    }

    results = []

    # 2. Tune XGBoost on the outer training rows only. Each CV clone fits a
    # fresh preprocessor on that fold's training partition.
    print("\n[4/5] Tuning XGBoost Model Hyperparameters...")
    xgb_pipeline = Pipeline([
        ('preprocessor', CupidFoldPreprocessor(target_col='compatible')),
        ('classifier', CupidXGBoost(n_jobs=-1, random_state=42).model)
    ])
    param_grid = {
        'classifier__n_estimators': [100, 150, 200],
        'classifier__max_depth': [4, 6, 8],
        'classifier__learning_rate': [0.03, 0.08, 0.12],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    best_xgb_sk, best_params, best_score, search = tune_hyperparameters(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        X_train=raw_train_df.drop(columns=['compatible']),
        y_train=y_train,
        method='random',
        n_iter=6,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        return_search=True
    )

    print(f"Best XGBoost Params: {best_params}")
    print(f"Best CV F1-Score: {best_score:.4f}")

    # Evaluate Tuned XGBoost
    # RandomizedSearchCV refits its best pipeline on all outer-training rows.
    # The fitted sklearn pipeline's fold preprocessing and refit are used for
    # evaluation; keep the separately persisted Flask objects in sync below.
    tuned_metrics = evaluate_classification_model(
        best_xgb_sk, raw_test_df.drop(columns=['compatible']), y_test,
        model_name="Tuned XGBoost"
    )
    print_evaluation_summary(tuned_metrics)

    # Compare the four original configurations on the same outer test set.
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_classification_model(model, X_test, y_test, model_name=name)
        print_evaluation_summary(metrics)
        results.append({key: metrics[key] for key in
                        ("Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")})
    results.append({key: tuned_metrics[key] for key in
                    ("Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")})

    results_df = pd.DataFrame(results)
    comparison_path = os.path.join(tables_dir, "model_comparison.csv")
    results_df.to_csv(comparison_path, index=False)
    # Keep the legacy top-level artifact current for existing consumers.
    results_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)
    tuning_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    tuning_df.to_csv(os.path.join(tables_dir, "tuning_results.csv"), index=False)

    # Save fold-wise F1 scores as true CV results, separate from test metrics.
    cv_rows = []
    for cv_name, cv_model, cv_params in [
        ("Logistic Regression", CupidLogisticRegression, {"max_iter": 1000, "random_state": 42}),
        ("Decision Tree", CupidDecisionTree, {"max_depth": 12, "min_samples_split": 10, "random_state": 42}),
        ("Random Forest", CupidRandomForest, {"n_estimators": 100, "max_depth": 14, "n_jobs": -1, "random_state": 42}),
        ("XGBoost", CupidXGBoost, {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.08, "n_jobs": -1, "random_state": 42}),
    ]:
        cv_summary, cv_scores = run_cross_validation(
            cv_model, cv_params, raw_train_df, n_splits=3, random_state=42, raw_data=True
        )
        for fold, score in enumerate(cv_scores["F1-Score"], start=1):
            cv_rows.append({"Model": cv_name, "Fold": fold, "F1-Score": float(score)})
        cv_rows.extend([
            {"Model": cv_name, "Fold": "Mean", "F1-Score": float(cv_summary["mean_F1-Score"])},
            {"Model": cv_name, "Fold": "Std", "F1-Score": float(cv_summary["std_F1-Score"])},
        ])
    cv_rows.append({"Model": "Tuned XGBoost (RandomizedSearchCV)", "Fold": "Mean", "F1-Score": float(best_score)})
    pd.DataFrame(cv_rows).to_csv(os.path.join(tables_dir, "cross_validation.csv"), index=False)

    # 3. Persist matching Flask wrapper artifacts trained on the full outer train.
    print("\n[5/5] Saving Best Model...")
    preprocessor_path = os.path.join(results_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    best_model_path = os.path.join(results_dir, "best_model.pkl")
    tuned_xgb = CupidXGBoost(
        n_estimators=best_params.get('classifier__n_estimators', 150),
        max_depth=best_params.get('classifier__max_depth', 6),
        learning_rate=best_params.get('classifier__learning_rate', 0.08),
        subsample=best_params.get('classifier__subsample', 0.8),
        colsample_bytree=best_params.get('classifier__colsample_bytree', 0.8),
        n_jobs=-1,
        random_state=42
    )
    tuned_xgb.fit(X_train, y_train)
    tuned_xgb.save(best_model_path)
    print(f"Successfully saved Best Model (Tuned XGBoost) to '{best_model_path}'")

    summary_file = os.path.join(results_dir, "summary.json")
    summary = {
            "best_model": "Tuned XGBoost",
            "best_params": {key.replace('classifier__', ''): value for key, value in best_params.items()},
            "best_cv_score": float(best_score),
            "evaluation": {
                "model_artifact": best_model_path,
                "preprocessor_artifact": preprocessor_path,
                "test_rows": int(len(y_test)),
                "test_size": 0.2,
                "random_state": 42,
                "scoring": "f1",
            },
            "metrics": {
                key: (None if pd.isna(tuned_metrics[key]) else float(tuned_metrics[key]))
                for key in ("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC")
            }
        }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, allow_nan=False)
    
    print("\nTraining and Evaluation pipeline complete!")

if __name__ == "__main__":
    main()
