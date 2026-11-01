import numpy as np
import pandas as pd

from src.data.preprocessing import CupidDataPreprocessor
from src.models import BaseModel, CupidXGBoost


def _make_dataset(n_rows=30):
    """Return a small, balanced dataset with all expected input columns."""
    data = {
        "compatible": np.array([0, 1] * (n_rows // 2)),
        "a_location": ["Paris", "Lyon", "Nice"] * (n_rows // 3),
        "b_location": ["Lyon", "Nice", "Paris"] * (n_rows // 3),
        "a_career_field": ["Engineering", "Arts", "Science"] * (n_rows // 3),
        "b_career_field": ["Arts", "Science", "Engineering"] * (n_rows // 3),
        "a_love_language": ["Words", "Time", "Acts"] * (n_rows // 3),
        "b_love_language": ["Time", "Acts", "Words"] * (n_rows // 3),
    }
    for column in CupidDataPreprocessor().numerical_raw_cols:
        data[column] = np.linspace(1.0, 10.0, n_rows)
    return pd.DataFrame(data)


def test_saved_model_and_preprocessor_work_together(tmp_path):
    dataset = _make_dataset()
    preprocessor = CupidDataPreprocessor()
    X, y = preprocessor.fit_transform(dataset)

    model = CupidXGBoost(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.2,
        n_jobs=1,
        random_state=42,
    )
    model.fit(X, y)

    preprocessor_path = tmp_path / "preprocessor.pkl"
    model_path = tmp_path / "model.pkl"
    preprocessor.save(str(preprocessor_path))
    model.save(str(model_path))

    # Follow the same reload pattern used by the Flask application.
    loaded_preprocessor = CupidDataPreprocessor.load(str(preprocessor_path))
    loaded_model = BaseModel.load(str(model_path))

    new_observation = dataset.drop(columns="compatible").iloc[[0]].copy()
    transformed_observation = loaded_preprocessor.transform(new_observation)
    prediction = loaded_model.predict(transformed_observation)
    probabilities = loaded_model.predict_proba(transformed_observation)

    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()
