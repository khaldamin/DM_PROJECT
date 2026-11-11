"""
SHAP Explainability Module for Cupid Matchmaking Models.
Generates local and global feature attribution explanations for predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from src.models.base_model import BaseModel
from src.data.preprocessing import CupidDataPreprocessor


class CupidExplainer:
    """
    SHAP-based explainer for Cupid matchmaking predictions.
    Computes global feature importance summary and local waterfall / force explanations.
    """
    def __init__(self, model_path: str = "results/best_model.pkl", preprocessor_path: str = "results/preprocessor.pkl"):
        self.model_wrapper = BaseModel.load(model_path)
        self.model = self.model_wrapper.model if hasattr(self.model_wrapper, 'model') else self.model_wrapper
        self.preprocessor = CupidDataPreprocessor.load(preprocessor_path)
        self.explainer = None
        self.feature_names = self.preprocessor.feature_names

    def initialize_explainer(self, background_data: np.ndarray):
        """Initializes TreeExplainer with background data."""
        self.explainer = shap.TreeExplainer(self.model, data=background_data)
        return self

    def explain_global(self, X_sample: np.ndarray, output_path: str = "results/figures/shap_summary.png"):
        """Generates SHAP beeswarm / summary plot across test samples."""
        if self.explainer is None:
            self.initialize_explainer(X_sample[:100])

        shap_values = self.explainer(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False, max_display=15)
        plt.title("SHAP Global Feature Importance (Impact on Compatibility)", fontsize=13, fontweight='bold', pad=15)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def explain_local_instance(self, df_pair: pd.DataFrame, output_path: str = "results/figures/shap_waterfall.png") -> dict:
        """
        Explains why a specific profile pair was predicted compatible or incompatible.
        Returns top positive drivers and top negative friction points.
        """
        X_trans = self.preprocessor.transform(df_pair)
        if self.explainer is None:
            self.initialize_explainer(X_trans)

        shap_values = self.explainer(X_trans)
        
        # Values for the first (and only) instance
        instance_shap = shap_values[0]
        base_value = float(instance_shap.base_values) if hasattr(instance_shap, 'base_values') else 0.0
        
        # Generate waterfall plot
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(instance_shap, max_display=10, show=False)
        plt.title("Local SHAP Explanation: Compatibility Decision Drivers", fontsize=12, fontweight='bold', pad=15)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Parse key contributors safely
        contributions = []
        n_features = min(len(self.feature_names), len(instance_shap.values))
        for name, val in zip(self.feature_names[:n_features], instance_shap.values[:n_features]):
            contributions.append({
                "feature": name,
                "shap_value": float(round(val, 4)),
                "impact": "Promotes Compatibility" if val > 0 else "Friction / Lowers Compatibility"
            })
        
        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "base_value": base_value,
            "prediction_margin": float(np.sum(instance_shap.values)),
            "top_drivers": contributions[:5],
            "waterfall_plot": output_path
        }
