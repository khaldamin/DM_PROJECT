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


class CupidFoldPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn adapter that fits Cupid preprocessing on each CV fold."""

    def __init__(self, target_col='compatible'):
        self.target_col = target_col

    def fit(self, X, y=None):
        self.preprocessor_ = CupidDataPreprocessor(target_col=self.target_col)
        self.preprocessor_.fit_transform(X)
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
    os.makedirs(results_dir, exist_ok=True)

    # 1. Preprocessing & Feature Engineering
    print("\n[1/5] Loading data and performing Feature Engineering...")
    preprocessor = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test(data_path)
    
    # Save preprocessor
    preprocessor_path = os.path.join(results_dir, "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    print(f"Preprocessed train shape: {X_train.shape}, test shape: {X_test.shape}")
    print(f"Saved preprocessor pipeline to '{preprocessor_path}'")

    # 2. Model Dictionary
    models = {
        "Logistic Regression": CupidLogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": CupidDecisionTree(max_depth=12, min_samples_split=10, random_state=42),
        "Random Forest": CupidRandomForest(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42),
        "XGBoost": CupidXGBoost(n_estimators=150, max_depth=6, learning_rate=0.08, n_jobs=-1, random_state=42)
    }

    results = []
    trained_models = {}

    # 3. Train & Evaluate Models
    print("\n[2/5] Training models...")
    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_classification_model(model, X_test, y_test, model_name=name)
        print_evaluation_summary(metrics)
        
        results.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-Score": metrics["F1-Score"],
            "ROC-AUC": metrics["ROC-AUC"]
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n[3/5] Initial Model Comparison Results:")
    print(results_df.to_string(index=False))

    # Save initial evaluation table
    results_df.to_csv(os.path.join(results_dir, "model_comparison.csv"), index=False)

    # 4. Hyperparameter Tuning for XGBoost
    print("\n[4/5] Tuning XGBoost Model Hyperparameters...")
    # Tune on the raw outer-training rows. Pipeline is cloned for every search
    # fold, so each cloned preprocessor learns only from that fold's training
    # partition. Keep the final test set out of the search entirely.
    raw_df = pd.read_csv(data_path)
    raw_train_df, _ = train_test_split(
        raw_df,
        test_size=0.2,
        random_state=42,
        stratify=raw_df['compatible']
    )
    raw_test_df = raw_df.drop(index=raw_train_df.index)
    X_raw_train = raw_train_df.drop(columns=['compatible'])
    y_raw_train = raw_train_df['compatible'].to_numpy()

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
    
    best_xgb_sk, best_params, best_score = tune_hyperparameters(
        estimator=xgb_pipeline,
        param_grid=param_grid,
        X_train=X_raw_train,
        y_train=y_raw_train,
        method='random',
        n_iter=6,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        random_state=42
    )

    print(f"Best XGBoost Params: {best_params}")
    print(f"Best CV F1-Score: {best_score:.4f}")

    # Evaluate Tuned XGBoost
    # RandomizedSearchCV refits its best pipeline on all outer-training rows.
    tuned_metrics = evaluate_classification_model(
        best_xgb_sk, raw_test_df.drop(columns=['compatible']),
        raw_test_df['compatible'].to_numpy(), model_name="Tuned XGBoost"
    )
    print_evaluation_summary(tuned_metrics)

    # 5. Save Best Model
    print("\n[5/5] Saving Best Model...")
    best_model_path = os.path.join(results_dir, "best_model.pkl")
    # Persist the existing wrapper format used by the application. Refit its
    # classifier on the outer training matrix produced by the same train-only
    # preprocessor saved above.
    tuned_xgb = CupidXGBoost(
        n_estimators=best_params['classifier__n_estimators'],
        max_depth=best_params['classifier__max_depth'],
        learning_rate=best_params['classifier__learning_rate'],
        subsample=best_params['classifier__subsample'],
        colsample_bytree=best_params['classifier__colsample_bytree'],
        n_jobs=-1,
        random_state=42
    )
    tuned_xgb.fit(X_train, y_train)
    tuned_xgb.save(best_model_path)
    print(f"Successfully saved Best Model (Tuned XGBoost) to '{best_model_path}'")

    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "best_model": "Tuned XGBoost",
            "best_params": {key.replace('classifier__', ''): value for key, value in best_params.items()},
            "metrics": {
                "Accuracy": float(tuned_metrics["Accuracy"]),
                "Precision": float(tuned_metrics["Precision"]),
                "Recall": float(tuned_metrics["Recall"]),
                "F1-Score": float(tuned_metrics["F1-Score"]),
                "ROC-AUC": float(tuned_metrics["ROC-AUC"])
            }
        }, f, indent=4)
    
    print("\nTraining and Evaluation pipeline complete!")

if __name__ == "__main__":
    main()
