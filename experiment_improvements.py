"""
experiment_improvements.py
--------------------------
Standalone experimentation script to improve model accuracy and performance.
NOTE: This script does NOT modify any existing project files.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.abspath('.'))
from src.data.preprocessing import CupidDataPreprocessor
from src.models import CupidLogisticRegression, CupidRandomForest, CupidXGBoost

def compute_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_val = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc_val, 4)
    }

def main():
    print("=" * 65)
    print(" Cupid Matchmaking — Benchmark of Model Improvement Strategies")
    print("=" * 65)

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    raw_df = pd.read_csv(data_path)

    # Identical stratified 80/20 train/test split as existing baseline
    raw_train_df, raw_test_df = train_test_split(
        raw_df, test_size=0.2, random_state=42, stratify=raw_df['compatible']
    )

    y_train = raw_train_df['compatible'].to_numpy()
    y_test = raw_test_df['compatible'].to_numpy()
    y_score_train = raw_train_df['compatibility_score'].to_numpy()
    y_score_test = raw_test_df['compatibility_score'].to_numpy()

    # Preprocessing with baseline preprocessor
    preprocessor = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_train, _ = preprocessor.fit_transform(raw_train_df)
    X_test = preprocessor.transform(raw_test_df)

    # Further split X_train for validation threshold tuning (e.g. 80/20 of train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    results = []

    # -------------------------------------------------------------
    # 1. Baseline: Tuned XGBoost (Current Production Model)
    # -------------------------------------------------------------
    print("\n[Strategy 1] Baseline Tuned XGBoost (Threshold = 0.50)...")
    base_xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.12,
        subsample=0.9, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        eval_metric="logloss"
    )
    base_xgb.fit(X_train, y_train)
    y_prob_base = base_xgb.predict_proba(X_test)[:, 1]
    y_pred_base = (y_prob_base >= 0.5).astype(int)
    m_base = compute_metrics(y_test, y_pred_base, y_prob_base)
    results.append({"Approach": "1. Baseline Tuned XGBoost (th=0.50)", **m_base})
    print(f"   Accuracy: {m_base['Accuracy']} | F1: {m_base['F1-Score']} | ROC-AUC: {m_base['ROC-AUC']}")

    # -------------------------------------------------------------
    # 2. Threshold Optimization on Validation Set
    # -------------------------------------------------------------
    print("\n[Strategy 2] Threshold Tuning for Maximum Accuracy...")
    base_xgb_val = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.12,
        subsample=0.9, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        eval_metric="logloss"
    )
    base_xgb_val.fit(X_tr, y_tr)
    val_probs = base_xgb_val.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.40, 0.60, 41)
    best_th = 0.5
    best_val_acc = 0
    for th in thresholds:
        acc = accuracy_score(y_val, (val_probs >= th).astype(int))
        if acc > best_val_acc:
            best_val_acc = acc
            best_th = th

    print(f"   Optimal Threshold found on validation set: {best_th:.3f}")
    y_pred_tuned_th = (y_prob_base >= best_th).astype(int)
    m_th = compute_metrics(y_test, y_pred_tuned_th, y_prob_base)
    results.append({"Approach": f"2. Threshold Tuned XGBoost (th={best_th:.2f})", **m_th})
    print(f"   Accuracy: {m_th['Accuracy']} | F1: {m_th['F1-Score']} | ROC-AUC: {m_th['ROC-AUC']}")

    # -------------------------------------------------------------
    # 3. XGBoost Regressor (Continuous Score Prediction -> Threshold >= 65)
    # -------------------------------------------------------------
    print("\n[Strategy 3] XGBoost Regressor (Learning Continuous Score >= 65)...")
    xgb_reg = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.10,
        subsample=0.9, colsample_bytree=0.8, n_jobs=-1, random_state=42
    )
    xgb_reg.fit(X_train, y_score_train)
    pred_scores_test = xgb_reg.predict(X_test)
    y_pred_reg = (pred_scores_test >= 65.0).astype(int)
    # Normalize predicted score into pseudo-probability for AUC computation
    pseudo_prob_reg = (pred_scores_test - pred_scores_test.min()) / (pred_scores_test.max() - pred_scores_test.min())
    m_reg = compute_metrics(y_test, y_pred_reg, pseudo_prob_reg)
    results.append({"Approach": "3. XGBoost Regressor on Continuous Score", **m_reg})
    print(f"   Accuracy: {m_reg['Accuracy']} | F1: {m_reg['F1-Score']} | ROC-AUC: {m_reg['ROC-AUC']}")

    # -------------------------------------------------------------
    # 4. Soft Voting Ensemble (XGBoost + Random Forest + Logistic Regression)
    # -------------------------------------------------------------
    print("\n[Strategy 4] Soft Voting Ensemble (XGB + RF + LogReg)...")
    rf_clf = CupidRandomForest(n_estimators=100, max_depth=14, n_jobs=-1, random_state=42).model
    lr_clf = CupidLogisticRegression(max_iter=1000, random_state=42).model

    voting_ensemble = VotingClassifier(
        estimators=[
            ('xgb', base_xgb),
            ('rf', rf_clf),
            ('lr', lr_clf)
        ],
        voting='soft',
        weights=[3, 1, 1]
    )
    voting_ensemble.fit(X_train, y_train)
    y_prob_vote = voting_ensemble.predict_proba(X_test)[:, 1]
    y_pred_vote = (y_prob_vote >= 0.5).astype(int)
    m_vote = compute_metrics(y_test, y_pred_vote, y_prob_vote)
    results.append({"Approach": "4. Weighted Soft Voting Ensemble", **m_vote})
    print(f"   Accuracy: {m_vote['Accuracy']} | F1: {m_vote['F1-Score']} | ROC-AUC: {m_vote['ROC-AUC']}")

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print(" FINAL COMPARISON SUMMARY")
    print("=" * 65)
    print(df_res.to_string(index=False))

    output_path = os.path.join("results", "tables", "experiment_improvements_comparison.csv")
    df_res.to_csv(output_path, index=False)
    print(f"\nSaved experimental results to '{output_path}'")

if __name__ == "__main__":
    main()

