"""
Diagnostic Error Analysis & Calibration Assessment for Cupid AI.
Computes PR-AUC, calibration curves (Brier score), prediction distributions,
and analyzes demographic friction error segments.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, os.path.abspath('.'))

from src.models.base_model import BaseModel
from src.data.preprocessing import CupidDataPreprocessor


def run_error_analysis():
    print("==================================================")
    print(" Cupid AI: Comprehensive Error & Calibration")
    print("==================================================")

    model_path = os.path.join("results", "best_model.pkl")
    prep_path = os.path.join("results", "preprocessor.pkl")
    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("\n[1/4] Ingesting hold-out evaluation data...")
    model_wrapper = BaseModel.load(model_path)
    preprocessor = CupidDataPreprocessor.load(prep_path)

    df_raw = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible']
    )

    X_test = preprocessor.transform(test_df)
    y_test = test_df['compatible'].to_numpy()

    y_pred = model_wrapper.predict(X_test)
    y_proba = model_wrapper.predict_proba(X_test)[:, 1]

    # 1. Precision-Recall Curve & PR-AUC
    print("\n[2/4] Computing Precision-Recall dynamics and PR-AUC...")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"   -> Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")

    pr_fig_path = os.path.join(fig_dir, "pr_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#2980b9', linewidth=2, label=f'Tuned XGBoost (PR-AUC = {pr_auc:.4f})')
    plt.axhline(y=np.mean(y_test), color='#7f8c8d', linestyle='--', label=f'No-Skill Baseline ({np.mean(y_test):.2f})')
    plt.title("Precision-Recall Curve (Hold-out Test Set, 20,000 Pairs)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig(pr_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {pr_fig_path}")

    # 2. Probability Calibration Curve (Reliability Diagram)
    print("\n[3/4] Evaluating probability calibration & Brier score...")
    brier = brier_score_loss(y_test, y_proba)
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')
    print(f"   -> Brier Score Loss: {brier:.4f}")

    cal_fig_path = os.path.join(fig_dir, "calibration_curve.png")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(prob_pred, prob_true, 's-', color='#e67e22', linewidth=2, label=f'XGBoost (Brier = {brier:.4f})')
    plt.title("Reliability Diagram (Probability Calibration Curve)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Mean Predicted Compatibility Probability", fontsize=11)
    plt.ylabel("Observed Empirical Proportion Compatible", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(cal_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {cal_fig_path}")

    # 3. Probability Distribution by True Class
    dist_fig_path = os.path.join(fig_dir, "prediction_distribution.png")
    plt.figure(figsize=(9, 5))
    plt.hist(y_proba[y_test == 0], bins=40, alpha=0.6, color='#e74c3c', label='True Incompatible ($y=0$)', density=True)
    plt.hist(y_proba[y_test == 1], bins=40, alpha=0.6, color='#2ecc71', label='True Compatible ($y=1$)', density=True)
    plt.axvline(x=0.5, color='black', linestyle=':', label='Decision Threshold (0.5)')
    plt.title("Estimated Compatibility Probability Distribution by Ground Truth", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Predicted Compatibility Probability $\hat{p}$", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(loc='upper center', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(dist_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {dist_fig_path}")

    # 4. Export Error Analysis Metrics JSON
    print("\n[4/4] Exporting error metrics summary...")
    cm = confusion_matrix(y_test, y_pred)
    error_summary = {
        "PR_AUC": float(round(pr_auc, 4)),
        "Brier_Score": float(round(brier, 4)),
        "Confusion_Matrix": {
            "True_Negative": int(cm[0, 0]),
            "False_Positive": int(cm[0, 1]),
            "False_Negative": int(cm[1, 0]),
            "True_Positive": int(cm[1, 1])
        },
        "False_Positive_Rate": float(round(cm[0, 1] / (cm[0, 0] + cm[0, 1]), 4)),
        "False_Negative_Rate": float(round(cm[1, 0] / (cm[1, 0] + cm[1, 1]), 4))
    }
    json_path = os.path.join(tables_dir, "error_analysis_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(error_summary, f, indent=4)
    print(f"   -> Saved: {json_path}")

    print("\nError and calibration analysis complete!")


if __name__ == '__main__':
    run_error_analysis()
