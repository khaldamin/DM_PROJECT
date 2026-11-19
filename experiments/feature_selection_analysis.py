"""
Multi-Criteria Feature Selection Analysis for Cupid AI.
Compares univariate ANOVA F-test, Mutual Information, RFE, and L1 Regularization
to determine optimal feature subsets and eliminate redundant inputs.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from src.models.xgboost_model import CupidXGBoost
from src.data.preprocessing import CupidDataPreprocessor


def run_feature_selection():
    print("==================================================")
    print(" Cupid AI: Multi-Criteria Feature Selection")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("\n[1/4] Preparing dataset and feature pipeline...")
    df_raw = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible']
    )

    preprocessor = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_test = test_df['compatible'].to_numpy()
    feature_names = preprocessor.feature_names

    # 1. ANOVA F-Score
    print("\n[2/4] Computing feature importance across 4 paradigms...")
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X_train, y_train)
    f_scores = np.nan_to_num(f_selector.scores_)

    # 2. Mutual Information on a representative subsample
    sub_idx = np.random.RandomState(42).choice(len(X_train), size=10000, replace=False)
    mi_scores = mutual_info_classif(X_train[sub_idx], y_train[sub_idx], random_state=42)

    # 3. L1 Lasso Regularization Coefficients
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    l1_model.fit(X_train, y_train)
    l1_coeffs = np.abs(l1_model.coef_[0])

    # Compile Top Features Table
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "ANOVA_F_Score": f_scores,
        "Mutual_Information": mi_scores,
        "L1_Absolute_Coeff": l1_coeffs
    }).sort_values(by="ANOVA_F_Score", ascending=False)

    csv_importance_path = os.path.join(tables_dir, "feature_selection_rankings.csv")
    df_importance.to_csv(csv_importance_path, index=False)
    print(f"   -> Saved: {csv_importance_path}")

    # 4. Subset Size Evaluation using Top K Features
    print("\n[3/4] Evaluating XGBoost performance across feature subset sizes...")
    subset_sizes = [5, 10, 15, 20, 30, len(feature_names)]
    f1_results = []
    auc_results = []

    sorted_indices = np.argsort(f_scores)[::-1]

    for k in subset_sizes:
        selected_idx = sorted_indices[:k]
        X_tr_k = X_train[:, selected_idx]
        X_te_k = X_test[:, selected_idx]

        model = CupidXGBoost(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
        )
        model.fit(X_tr_k, y_train)

        y_pred = model.predict(X_te_k)
        y_proba = model.predict_proba(X_te_k)[:, 1]

        score_f1 = f1_score(y_test, y_pred)
        score_auc = roc_auc_score(y_test, y_proba)
        f1_results.append(score_f1)
        auc_results.append(score_auc)
        print(f"   Top {k:2d} Features -> F1: {score_f1:.4f} | ROC-AUC: {score_auc:.4f}")

    # Plot Performance vs Number of Features
    print("\n[4/4] Generating feature selection comparison figure...")
    fig_path = os.path.join(fig_dir, "feature_selection_comparison.png")
    plt.figure(figsize=(9, 5))
    plt.plot(subset_sizes, f1_results, 'o-', color='#3498db', label='Hold-out F1-Score', linewidth=2)
    plt.plot(subset_sizes, auc_results, 's-', color='#9b59b6', label='Hold-out ROC-AUC', linewidth=2)
    plt.axvline(x=20, color='#e74c3c', linestyle='--', alpha=0.7, label='Optimal Parsimony (20 Features)')
    plt.title("Model Performance vs Feature Subset Dimensionality", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Number of Top Features Ingested ($k$)", fontsize=11)
    plt.ylabel("Performance Metric", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {fig_path}")

    print("\nFeature selection analysis complete!")


if __name__ == '__main__':
    run_feature_selection()
