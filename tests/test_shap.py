import numpy as np
import pandas as pd
import pytest
import os

from src.evaluation.shap_explainer import CupidExplainer

def test_shap_explainer_initialization():
    explainer = CupidExplainer()
    assert explainer.model is not None
    assert explainer.preprocessor is not None
    assert len(explainer.feature_names) > 0

def test_shap_local_instance_explanation():
    data_path = os.path.join("data", "raw", "cupid_algorithm_dataset.csv")
    df = pd.read_csv(data_path, nrows=5)
    
    explainer = CupidExplainer()
    result = explainer.explain_local_instance(df.iloc[[0]], output_path="results/figures/test_shap_waterfall.png")
    
    assert "top_drivers" in result
    assert len(result["top_drivers"]) == 5
    assert "waterfall_plot" in result
    assert os.path.exists("results/figures/test_shap_waterfall.png")
    
    if os.path.exists("results/figures/test_shap_waterfall.png"):
        os.remove("results/figures/test_shap_waterfall.png")
