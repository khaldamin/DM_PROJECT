"""
Script to compute SHAP Global & Local Figures.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from src.evaluation.shap_explainer import CupidExplainer

def run_shap_generation():
    print("=" * 65)
    print("  GENERATING SHAP EXPLAINABILITY FIGURES (GLOBAL & LOCAL)")
    print("=" * 65)

    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    df = pd.read_csv(data_path)

    explainer = CupidExplainer()

    # Preprocess a sample of 200 pairs for global SHAP summary
    print("[1/2] Computing Global SHAP Summary Plot...")
    sample_df = df.drop(columns=['compatible', 'pair_id', 'compatibility_score', 'relationship_longevity_months'], errors='ignore').iloc[:200]
    X_sample = explainer.preprocessor.transform(sample_df)

    global_fig = explainer.explain_global(X_sample, output_path="results/figures/shap_summary.png")
    print(f"-> Saved Global SHAP Summary to {global_fig}")

    # Local waterfall explanation for a specific couple
    print("[2/2] Computing Local SHAP Waterfall Explanation for Sample Profile Pair...")
    pair_df = df.iloc[[0]]
    local_explanation = explainer.explain_local_instance(pair_df, output_path="results/figures/shap_waterfall.png")
    print(f"-> Saved Local SHAP Waterfall to {local_explanation['waterfall_plot']}")
    print("Top local feature drivers:")
    for driver in local_explanation['top_drivers']:
        print(f"  * {driver['feature']}: {driver['shap_value']} ({driver['impact']})")

    print("\nSHAP explainability figures generated successfully!")

if __name__ == "__main__":
    run_shap_generation()

