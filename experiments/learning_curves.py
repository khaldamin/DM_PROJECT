"""
Learning Curves & Bias-Variance Analysis for Cupid AI.
Evaluates training and validation F1-Score & ROC-AUC across dataset sample sizes
to diagnose sample efficiency and model complexity.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from src.models.xgboost_model import CupidXGBoost
from src.data.preprocessing import CupidDataPreprocessor


def compute_learning_curves():
    print("==================================================")
    print(" Cupid AI: Learning Curves & Sample Efficiency")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("\n[1/3] Loading data and preprocessing...")
    df_raw = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible']
    )

    preprocessor = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    y_test = test_df['compatible'].to_numpy()

    # Fractions of training data to evaluate
    train_sizes = np.linspace(0.1, 1.0, 8)
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_scores_mean = []
    train_scores_std = []
    val_scores_mean = []
    val_scores_std = []
    sample_counts = []

    print("\n[2/3] Evaluating model across training size increments...")
    for frac in train_sizes:
        n_samples = int(frac * len(X_train))
        sample_counts.append(n_samples)
        fold_train_f1 = []
        fold_val_f1 = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            # Select subset of training fold
            sub_train_idx = train_idx[:int(frac * len(train_idx))]
            X_tr_sub, y_tr_sub = X_train[sub_train_idx], y_train[sub_train_idx]
            X_va, y_va = X_train[val_idx], y_train[val_idx]

            # Fast evaluation using baseline parameters
            model = CupidXGBoost(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
            )
            model.fit(X_tr_sub, y_tr_sub)

            y_tr_pred = model.predict(X_tr_sub)
            y_va_pred = model.predict(X_va)

            fold_train_f1.append(f1_score(y_tr_sub, y_tr_pred))
            fold_val_f1.append(f1_score(y_va, y_va_pred))

        mean_tr = float(np.mean(fold_train_f1))
        std_tr = float(np.std(fold_train_f1))
        mean_va = float(np.mean(fold_val_f1))
        std_va = float(np.std(fold_val_f1))

        train_scores_mean.append(mean_tr)
        train_scores_std.append(std_tr)
        val_scores_mean.append(mean_va)
        val_scores_std.append(std_va)

        print(f"   Fraction {frac*100:4.1f}% ({n_samples:,} pairs) -> Train F1: {mean_tr:.4f} | Val F1: {mean_va:.4f}")

    # Plot Learning Curve
    print("\n[3/3] Generating learning curves figure & table...")
    fig_path = os.path.join(fig_dir, "learning_curve_xgboost.png")
    plt.figure(figsize=(10, 6))
    
    plt.plot(sample_counts, train_scores_mean, 'o-', color='#e74c3c', label='Training F1-Score', linewidth=2)
    plt.fill_between(
        sample_counts,
        np.array(train_scores_mean) - np.array(train_scores_std),
        np.array(train_scores_mean) + np.array(train_scores_std),
        alpha=0.15, color='#e74c3c'
    )

    plt.plot(sample_counts, val_scores_mean, 's-', color='#2ecc71', label='Validation F1-Score', linewidth=2)
    plt.fill_between(
        sample_counts,
        np.array(val_scores_mean) - np.array(val_scores_std),
        np.array(val_scores_mean) + np.array(val_scores_std),
        alpha=0.15, color='#2ecc71'
    )

    plt.title("XGBoost Learning Curve: Bias-Variance & Sample Efficiency", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Training Observations ($N$)", fontsize=11)
    plt.ylabel("F1-Score", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {fig_path}")

    # Save CSV Data
    df_results = pd.DataFrame({
        "Training_Samples": sample_counts,
        "Fraction": train_sizes,
        "Train_F1_Mean": train_scores_mean,
        "Train_F1_Std": train_scores_std,
        "Val_F1_Mean": val_scores_mean,
        "Val_F1_Std": val_scores_std
    })
    csv_path = os.path.join(tables_dir, "learning_curves_data.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"   -> Saved: {csv_path}")

    print("\nLearning curves execution complete!")


if __name__ == '__main__':
    compute_learning_curves()
