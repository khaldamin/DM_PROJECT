import numpy as np
import pandas as pd

from src.data.preprocessing import CupidDataPreprocessor


def _make_dataset(n_rows=20):
    """Build a small dataset with all columns expected by the preprocessor."""
    data = {
        "compatible": np.array([0, 1] * (n_rows // 2)),
        "a_location": ["Paris"] * n_rows,
        "b_location": ["Lyon"] * n_rows,
        "a_career_field": ["Engineering"] * n_rows,
        "b_career_field": ["Arts"] * n_rows,
        "a_love_language": ["Words"] * n_rows,
        "b_love_language": ["Time"] * n_rows,
    }
    for col in CupidDataPreprocessor().numerical_raw_cols:
        data[col] = np.arange(n_rows, dtype=float)
    return pd.DataFrame(data)


def _run_with_dataframe(monkeypatch, preprocessor, df):
    monkeypatch.setattr(pd, "read_csv", lambda _path: df.copy())
    return preprocessor.prepare_train_test("in-memory.csv")


def test_prepare_train_test_fits_scaler_only_on_training_rows(monkeypatch):
    df = _make_dataset()
    preprocessor = CupidDataPreprocessor(test_size=0.25, random_state=11)

    X_train, X_test, y_train, y_test = _run_with_dataframe(monkeypatch, preprocessor, df)

    train_df, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        df,
        test_size=0.25,
        random_state=11,
        stratify=df["compatible"].values,
    )
    expected_mean = train_df["a_age"].mean()
    expected_scale = train_df["a_age"].std(ddof=0)
    scaler = preprocessor.encoder.named_transformers_["num"]

    assert scaler.mean_[0] == expected_mean
    assert scaler.scale_[0] == expected_scale
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_test_only_categories_are_ignored_and_dimensions_stay_compatible(monkeypatch):
    df = _make_dataset()
    # With this deterministic split, reserve a category for the test partition
    # by assigning it to rows the splitter selects for testing.
    _, test_indices = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        np.arange(len(df)),
        test_size=0.25,
        random_state=11,
        stratify=df["compatible"].values,
    )
    df.loc[test_indices[0], "a_location"] = "Test-only city"
    preprocessor = CupidDataPreprocessor(test_size=0.25, random_state=11)

    X_train, X_test, _, _ = _run_with_dataframe(monkeypatch, preprocessor, df)

    learned_categories = preprocessor.encoder.named_transformers_["cat"].categories_[0]
    assert "Test-only city" not in learned_categories
    assert X_train.shape[1] == X_test.shape[1] == len(preprocessor.feature_names)
    assert np.isfinite(X_test).all()


def test_split_does_not_call_fit_transform_on_full_dataset(monkeypatch):
    df = _make_dataset()
    preprocessor = CupidDataPreprocessor(test_size=0.25, random_state=11)
    original_fit_transform = preprocessor.fit_transform
    fitted_lengths = []

    def record_fit_transform(train_df):
        fitted_lengths.append(len(train_df))
        return original_fit_transform(train_df)

    monkeypatch.setattr(preprocessor, "fit_transform", record_fit_transform)

    X_train, X_test, _, _ = _run_with_dataframe(monkeypatch, preprocessor, df)

    assert fitted_lengths == [len(X_train)]
    assert len(X_train) + len(X_test) == len(df)
