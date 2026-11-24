import json
import os
import shutil
import uuid

import numpy as np
import pandas as pd

from src.train_and_evaluate import main
from src.data.preprocessing import CupidDataPreprocessor
from src.models import BaseModel


def _synthetic_data(n_rows=30):
    data = {"compatible": np.array([0, 1] * (n_rows // 2))}
    for column in CupidDataPreprocessor().categorical_cols:
        data[column] = ["A", "B", "C"] * (n_rows // 3)
    for column in CupidDataPreprocessor().numerical_raw_cols:
        data[column] = np.linspace(1, 10, n_rows)
    return pd.DataFrame(data)


def test_pipeline_writes_expected_artifacts_and_compatible_models(monkeypatch):
    import src.train_and_evaluate as training
    temp_dir = os.path.abspath(f".training-artifacts-{uuid.uuid4().hex}")
    results_dir = os.path.join(temp_dir, "results")
    original_read_csv = pd.read_csv
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    monkeypatch.setattr(pd, "read_csv", lambda _path: _synthetic_data())
    monkeypatch.chdir(temp_dir)
    original_tune = training.tune_hyperparameters

    def small_tune(**kwargs):
        kwargs["param_grid"] = {"classifier__max_depth": [2]}
        kwargs["n_iter"] = 1
        kwargs["n_jobs"] = 1
        return original_tune(**kwargs)

    monkeypatch.setattr(training, "tune_hyperparameters", small_tune)
    monkeypatch.setattr(training, "CupidRandomForest", lambda **kw: __import__(
            "src.models", fromlist=["CupidDecisionTree"]
        ).CupidDecisionTree(max_depth=2, random_state=42))

    try:
        main()

        comparison = original_read_csv(os.path.join(results_dir, "tables", "model_comparison.csv"))
        assert comparison["Model"].tolist() == [
            "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Tuned XGBoost"
        ]
        assert os.path.getsize(os.path.join(results_dir, "tables", "tuning_results.csv")) > 0
        tuning = original_read_csv(os.path.join(results_dir, "tables", "tuning_results.csv"))
        assert {"mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time", "mean_score_time"}.issubset(tuning.columns)
        assert tuning["rank_test_score"].is_monotonic_increasing
        cv = original_read_csv(os.path.join(results_dir, "tables", "cross_validation.csv"))
        assert {"Model", "Fold", "F1-Score"}.issubset(cv.columns)
        assert {"Mean", "Std"}.issubset(set(cv["Fold"]))

        with open(os.path.join(results_dir, "summary.json"), encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        assert summary["best_model"] == "Tuned XGBoost"
        assert "best_cv_score" in summary
        preprocessor = CupidDataPreprocessor.load(os.path.join(results_dir, "preprocessor.pkl"))
        model = BaseModel.load(os.path.join(results_dir, "best_model.pkl"))
        raw_observation = _synthetic_data().drop(columns="compatible").iloc[[0]]
        transformed = preprocessor.transform(raw_observation)
        assert model.predict(transformed).shape == (1,)
        assert model.predict_proba(transformed).shape == (1, 2)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
