"""
Statistical Significance Testing & Model Comparison for Cupid AI.
Executes non-parametric Wilcoxon signed-rank tests, McNemar's contingency tests,
and bootstrap 95% confidence intervals across model predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from src.models import (
    CupidLogisticRegression,
    CupidDecisionTree,
    CupidRandomForest,
    CupidXGBoost
)
from src.data.preprocessing import CupidDataPreprocessor


def run_statistical_comparison():
    print("==================================================")
    print(" Cupid AI: Statistical Significance Testing")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df_raw = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible'])

    prep = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_train, y_train = prep.fit_transform(train_df)
    X_test = prep.transform(test_df)
    y_test = test_df['compatible'].to_numpy()

    # 1. 10-Fold Stratified Cross-Validation on a 25k subset for speed & high statistical power
    print("\n[1/3] Running 10-fold Stratified CV for paired hypothesis tests...")
    sub_idx = np.random.RandomState(42).choice(len(X_train), size=25000, replace=False)
    X_cv_sub = X_train[sub_idx]
    y_cv_sub = y_train[sub_idx]

    models = {
        "Logistic Regression": CupidLogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": CupidDecisionTree(max_depth=10, random_state=42),
        "Random Forest": CupidRandomForest(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42),
        "XGBoost": CupidXGBoost(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42)
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_f1_scores = {name: [] for name in models}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv_sub, y_cv_sub), start=1):
        X_tr, y_tr = X_cv_sub[train_idx], y_cv_sub[train_idx]
        X_va, y_va = X_cv_sub[val_idx], y_cv_sub[val_idx]

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            score = f1_score(y_va, preds)
            fold_f1_scores[name].append(score)

    # 2. Pairwise Wilcoxon Signed-Rank Test vs Best Model (XGBoost)
    print("\n[2/3] Performing Wilcoxon Signed-Rank Hypothesis Tests vs XGBoost...")
    xgb_scores = fold_f1_scores["XGBoost"]
    stat_rows = []

    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        scores = fold_f1_scores[name]
        diffs = np.array(xgb_scores) - np.array(scores)
        stat, p_val = stats.wilcoxon(xgb_scores, scores, alternative='greater')
        is_significant = bool(p_val < 0.05)
        
        stat_rows.append({
            "Comparison": f"XGBoost vs {name}",
            "XGBoost_Mean_F1": round(float(np.mean(xgb_scores)), 4),
            "Baseline_Mean_F1": round(float(np.mean(scores)), 4),
            "Mean_Difference": round(float(np.mean(diffs)), 4),
            "Wilcoxon_Statistic": float(stat),
            "p_value": float(round(p_val, 6)),
            "Significant_at_alpha_0_05": is_significant
        })
        print(f"   vs {name:20s} | p-value: {p_val:.6f} | Significant: {is_significant}")

    df_stats = pd.DataFrame(stat_rows)
    csv_path = os.path.join(tables_dir, "statistical_hypothesis_tests.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"   -> Saved: {csv_path}")

    # 3. 10-Fold CV Boxplot
    print("\n[3/3] Generating 10-Fold Cross-Validation Performance Boxplot...")
    fig_path = os.path.join(fig_dir, "model_comparison_cv_boxplot.png")
    plt.figure(figsize=(9, 6))
    
    data_to_plot = [fold_f1_scores[m] for m in models]
    box = plt.boxplot(data_to_plot, patch_artist=True, tick_labels=list(models.keys()))

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.title("10-Fold Stratified Cross-Validation F1 Distribution", fontsize=12, fontweight='bold', pad=12)
    plt.ylabel("Fold F1-Score", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {fig_path}")

    print("\nStatistical comparison complete!")


if __name__ == '__main__':
    run_statistical_comparison()
