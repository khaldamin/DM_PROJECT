import os
import sys
import json
import pandas as pd
import numpy as np

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
    xgb_base = CupidXGBoost(n_jobs=-1, random_state=42).model
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.03, 0.08, 0.12],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    best_xgb_sk, best_params, best_score = tune_hyperparameters(
        estimator=xgb_base,
        param_grid=param_grid,
        X_train=X_train,
        y_train=y_train,
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
    tuned_xgb = CupidXGBoost(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        n_jobs=-1,
        random_state=42
    )
    tuned_xgb.fit(X_train, y_train)
    tuned_metrics = evaluate_classification_model(tuned_xgb, X_test, y_test, model_name="Tuned XGBoost")
    print_evaluation_summary(tuned_metrics)

    # 5. Save Best Model
    print("\n[5/5] Saving Best Model...")
    best_model_path = os.path.join(results_dir, "best_model.pkl")
    tuned_xgb.save(best_model_path)
    print(f"Successfully saved Best Model (Tuned XGBoost) to '{best_model_path}'")

    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "best_model": "Tuned XGBoost",
            "best_params": best_params,
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
