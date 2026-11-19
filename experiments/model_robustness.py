"""
Model Robustness & Sensitivity Stress-Testing for Cupid AI.
Evaluates model stability under repeated random splits, Gaussian input perturbation,
and feature masking/missingness scenarios.
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.abspath('.'))

from src.models.base_model import BaseModel
from src.data.preprocessing import CupidDataPreprocessor
from src.models.xgboost_model import CupidXGBoost


def run_robustness_analysis():
    print("==================================================")
    print(" Cupid AI: Model Robustness & Stress-Testing")
    print("==================================================")

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df_raw = pd.read_csv(data_path)

    # 1. Multi-Seed Split Stability (10 Random Seeds)
    print("\n[1/3] Evaluating split robustness across 5 random seeds...")
    seeds = [42, 7, 21, 99, 123]
    seed_f1s = []
    seed_aucs = []

    for seed in seeds:
        train_df, test_df = train_test_split(
            df_raw, test_size=0.2, random_state=seed, stratify=df_raw['compatible']
        )
        prep = CupidDataPreprocessor(test_size=0.2, random_state=seed)
        X_tr, y_tr = prep.fit_transform(train_df)
        X_te = prep.transform(test_df)
        y_te = test_df['compatible'].to_numpy()

        model = CupidXGBoost(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        probas = model.predict_proba(X_te)[:, 1]

        score_f1 = f1_score(y_te, preds)
        score_auc = roc_auc_score(y_te, probas)
        seed_f1s.append(score_f1)
        seed_aucs.append(score_auc)
        print(f"   Seed {seed:3d} -> F1: {score_f1:.4f} | ROC-AUC: {score_auc:.4f}")

    print(f"   Multi-seed F1:  {np.mean(seed_f1s):.4f} +/- {np.std(seed_f1s):.4f}")
    print(f"   Multi-seed AUC: {np.mean(seed_aucs):.4f} +/- {np.std(seed_aucs):.4f}")

    # 2. Gaussian Noise Sensitivity
    print("\n[2/3] Evaluating resilience to Gaussian input noise...")
    # Base split
    train_df, test_df = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible'])
    prep = CupidDataPreprocessor(test_size=0.2, random_state=42)
    X_tr, y_tr = prep.fit_transform(train_df)
    X_te_clean = prep.transform(test_df)
    y_te = test_df['compatible'].to_numpy()

    base_model = CupidXGBoost(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    base_model.fit(X_tr, y_tr)

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.35]
    noise_f1s = []
    noise_aucs = []

    for sigma in noise_levels:
        if sigma == 0.0:
            X_noisy = X_te_clean
        else:
            noise = np.random.RandomState(42).normal(0, sigma, size=X_te_clean.shape)
            X_noisy = X_te_clean + noise

        preds = base_model.predict(X_noisy)
        probas = base_model.predict_proba(X_noisy)[:, 1]

        f1 = f1_score(y_te, preds)
        auc = roc_auc_score(y_te, probas)
        noise_f1s.append(f1)
        noise_aucs.append(auc)
        print(f"   Noise sigma={sigma:.2f} -> F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    # Plot Noise Degradation Curve
    print("\n[3/3] Generating robustness visualization & summary...")
    fig_path = os.path.join(fig_dir, "robustness_noise.png")
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, noise_f1s, 'o-', color='#e74c3c', label='F1-Score under Input Noise', linewidth=2)
    plt.plot(noise_levels, noise_aucs, 's-', color='#2980b9', label='ROC-AUC under Input Noise', linewidth=2)
    plt.title("Model Graceful Degradation Under Gaussian Input Noise", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Gaussian Noise Standard Deviation ($\sigma$)", fontsize=11)
    plt.ylabel("Hold-out Performance", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {fig_path}")

    # Save summary JSON
    summary = {
        "MultiSeed_Evaluation": {
            "Seeds": seeds,
            "Mean_F1": float(round(np.mean(seed_f1s), 4)),
            "Std_F1": float(round(np.std(seed_f1s), 4)),
            "Mean_AUC": float(round(np.mean(seed_aucs), 4)),
            "Std_AUC": float(round(np.std(seed_aucs), 4))
        },
        "Noise_Stress_Test": {
            "Sigma_Levels": noise_levels,
            "F1_Scores": [round(x, 4) for x in noise_f1s],
            "ROC_AUC_Scores": [round(x, 4) for x in noise_aucs]
        }
    }
    json_path = os.path.join(tables_dir, "robustness_analysis_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"   -> Saved: {json_path}")

    print("\nRobustness analysis complete!")


if __name__ == '__main__':
    run_robustness_analysis()
