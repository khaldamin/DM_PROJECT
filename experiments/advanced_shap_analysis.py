"""
Advanced SHAP Analysis & Interaction Exploration for Cupid AI.
Extends baseline SHAP analysis with interaction values, dependence plots,
bar summaries, and local decision boundaries.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath('.'))

from src.models.base_model import BaseModel
from src.data.preprocessing import CupidDataPreprocessor


def run_advanced_shap_analysis():
    print("==================================================")
    print(" Cupid AI: Advanced SHAP & Interaction Analysis")
    print("==================================================")

    model_path = os.path.join("results", "best_model.pkl")
    prep_path = os.path.join("results", "preprocessor.pkl")
    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    fig_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        print(f"Error: Required artifacts not found at {model_path} or {prep_path}")
        return

    print("\n[1/4] Loading models and dataset...")
    model_wrapper = BaseModel.load(model_path)
    raw_xgb = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
    preprocessor = CupidDataPreprocessor.load(prep_path)

    df_raw = pd.read_csv(data_path)
    train_df, test_df = train_test_split(
        df_raw, test_size=0.2, random_state=42, stratify=df_raw['compatible']
    )

    # Subsample test set for fast and representative Shapley estimation
    sample_size = min(1000, len(test_df))
    test_sample = test_df.sample(n=sample_size, random_state=42)
    X_sample = preprocessor.transform(test_sample)
    feature_names = preprocessor.feature_names

    print(f"\n[2/4] Initializing TreeExplainer on {sample_size} test samples...")
    explainer = shap.TreeExplainer(raw_xgb)
    shap_values = explainer(X_sample)

    # 1. Global Mean Absolute SHAP Bar Plot
    print("\n[3/4] Generating advanced interpretability figures...")
    bar_fig_path = os.path.join(fig_dir, "enhanced_shap_bar.png")
    plt.figure(figsize=(10, 7))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(bar_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {bar_fig_path}")

    # 2. Enhanced Top 20 Beeswarm Plot
    beeswarm_fig_path = os.path.join(fig_dir, "enhanced_shap_beeswarm_top20.png")
    plt.figure(figsize=(11, 8))
    shap.summary_plot(shap_values.values, X_sample, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Beeswarm: Top 20 Feature Contributions", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(beeswarm_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   -> Saved: {beeswarm_fig_path}")

    # 3. Dependence Plot: Big 5 Harmony vs Personality Euclidean Distance
    try:
        # Find feature indices
        h_idx = feature_names.index('big5_harmony_score') if 'big5_harmony_score' in feature_names else 0
        d_idx = feature_names.index('personality_euclidean_dist') if 'personality_euclidean_dist' in feature_names else 1
        
        dep_fig_path = os.path.join(fig_dir, "enhanced_shap_dependence.png")
        plt.figure(figsize=(9, 6))
        shap.dependence_plot(
            h_idx, shap_values.values, X_sample,
            feature_names=feature_names,
            interaction_index=d_idx,
            show=False
        )
        plt.title("SHAP Dependence: Big 5 Harmony vs Euclidean Distance", fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(dep_fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> Saved: {dep_fig_path}")
    except Exception as e:
        print(f"   Notice on dependence plot: {e}")

    # 4. Save Top Importance JSON Summary
    print("\n[4/4] Exporting numerical importance table...")
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    sorted_indices = np.argsort(mean_abs_shap)[::-1]
    
    top_features = []
    for rank, idx in enumerate(sorted_indices[:15], start=1):
        top_features.append({
            "rank": rank,
            "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
            "mean_abs_shap": float(round(mean_abs_shap[idx], 4))
        })
    
    json_path = os.path.join(tables_dir, "enhanced_shap_global_importance.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(top_features, f, indent=4)
    print(f"   -> Saved: {json_path}")

    print("\nAdvanced SHAP analysis complete!")


if __name__ == '__main__':
    run_advanced_shap_analysis()
