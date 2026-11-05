import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import src.evluation.cross_validation as cross_validation
from src.data.preprocessing import CupidDataPreprocessor
from src.models import CupidDecisionTree
from src.train_and_evaluate import CupidFoldPreprocessor
from src.evluation.tuning import tune_hyperparameters


def _make_raw_data(n_rows=30):
    data = {
        "compatible": np.array([0, 1] * (n_rows // 2)),
        "a_location": ["Paris"] * n_rows,
        "b_location": ["Lyon"] * n_rows,
        "a_career_field": ["Engineering"] * n_rows,
        "b_career_field": ["Arts"] * n_rows,
        "a_love_language": ["Words"] * n_rows,
        "b_love_language": ["Time"] * n_rows,
    }
    for column in CupidDataPreprocessor().numerical_raw_cols:
        data[column] = np.arange(n_rows, dtype=float)
    return pd.DataFrame(data)


def test_raw_cross_validation_fits_fresh_preprocessor_per_fold(monkeypatch):
    df = _make_raw_data()
    created_preprocessors = []
    created_models = []
    original_init = CupidDataPreprocessor.__init__
    original_fit_transform = CupidDataPreprocessor.fit_transform
    fit_rows = []

    def record_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created_preprocessors.append(self)

    def record_fit(self, train_df):
        fit_rows.append(train_df.index.to_numpy())
        return original_fit_transform(self, train_df)

    def model_factory(**kwargs):
        model = CupidDecisionTree(max_depth=2, **kwargs)
        created_models.append(model)
        return model

    monkeypatch.setattr(CupidDataPreprocessor, "__init__", record_init)
    monkeypatch.setattr(CupidDataPreprocessor, "fit_transform", record_fit)

    cross_validation.run_cross_validation(
        model_factory, {}, df, None, n_splits=3, raw_data=True
    )

    assert len(created_preprocessors) == len(created_models) == 3
    assert len({id(preprocessor) for preprocessor in created_preprocessors}) == 3
    assert len({id(model) for model in created_models}) == 3
    assert all(len(rows) == 20 for rows in fit_rows)


def test_raw_cross_validation_fits_scaler_and_encoder_only_on_fold_train(monkeypatch):
    df = _make_raw_data()
    # Assign categories only to the rows that this deterministic CV reserves
    # for validation in the first fold.
    _, validation_idx = next(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(df, df.compatible))
    df.loc[validation_idx[0], "a_location"] = "Validation-only city"
    preprocessor_instances = []
    original_init = CupidDataPreprocessor.__init__

    def record_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        preprocessor_instances.append(self)

    monkeypatch.setattr(CupidDataPreprocessor, "__init__", record_init)
    cross_validation.run_cross_validation(
        CupidDecisionTree, {"max_depth": 2}, df, None,
        n_splits=3, random_state=42, raw_data=True
    )

    first_preprocessor = preprocessor_instances[0]
    scaler = first_preprocessor.encoder.named_transformers_["num"]
    # First fold's training indices are the complement of validation_idx.
    expected_train = df.drop(index=validation_idx)
    assert scaler.mean_[0] == expected_train["a_age"].mean()
    learned_categories = first_preprocessor.encoder.named_transformers_["cat"].categories_[0]
    assert "Validation-only city" not in learned_categories


def test_raw_cross_validation_keeps_train_and_validation_dimensions_compatible(monkeypatch):
    df = _make_raw_data()
    shapes = []
    train_shapes = []
    original_fit_transform = CupidDataPreprocessor.fit_transform
    original_transform = CupidDataPreprocessor.transform

    def record_fit_transform(self, train_df):
        transformed, target = original_fit_transform(self, train_df)
        train_shapes.append(transformed.shape[1])
        return transformed, target

    def record_transform(self, validation_df):
        transformed = original_transform(self, validation_df)
        shapes.append(transformed.shape[1])
        return transformed

    monkeypatch.setattr(CupidDataPreprocessor, "fit_transform", record_fit_transform)
    monkeypatch.setattr(CupidDataPreprocessor, "transform", record_transform)
    cross_validation.run_cross_validation(
        CupidDecisionTree, {"max_depth": 2}, df, None,
        n_splits=3, raw_data=True
    )
    assert train_shapes == shapes


def test_randomized_tuning_refits_preprocessor_inside_each_fold(monkeypatch):
    df = _make_raw_data()
    X = df.drop(columns="compatible")
    y = df["compatible"].to_numpy()
    fitted_preprocessors = []
    fitted_means = []
    original_fit_transform = CupidDataPreprocessor.fit_transform

    def record_fit_transform(self, train_df):
        fitted_preprocessors.append(self)
        fitted_means.append(self.numerical_raw_cols and train_df["a_age"].mean())
        return original_fit_transform(self, train_df)

    monkeypatch.setattr(CupidDataPreprocessor, "fit_transform", record_fit_transform)
    estimator = Pipeline([
        ("preprocessor", CupidFoldPreprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=0)),
    ])

    _, _, _, search = tune_hyperparameters(
        estimator=estimator,
        param_grid={"classifier__max_depth": [1, 2]},
        X_train=X,
        y_train=y,
        method="random",
        n_iter=2,
        cv=3,
        scoring="f1",
        n_jobs=1,
        random_state=7,
        return_search=True,
    )

    # 3 folds per candidate plus one final refit on all supplied training rows.
    assert len(fitted_preprocessors) == 2 * 3 + 1
    assert len({id(preprocessor) for preprocessor in fitted_preprocessors}) == len(fitted_preprocessors)
    assert len(set(fitted_means[:3])) > 1
    assert {"mean_test_score", "std_test_score", "rank_test_score", "mean_fit_time", "mean_score_time"}.issubset(search.cv_results_)


def test_randomized_tuning_is_reproducible():
    df = _make_raw_data()
    X = df.drop(columns="compatible")
    y = df["compatible"].to_numpy()
    estimator = Pipeline([
        ("preprocessor", CupidFoldPreprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=0)),
    ])
    kwargs = dict(
        estimator=estimator,
        param_grid={"classifier__max_depth": [1, 2, 3]},
        X_train=X, y_train=y, method="random", n_iter=2, cv=3,
        scoring="f1", n_jobs=1, random_state=42, return_search=True,
    )
    first = tune_hyperparameters(**kwargs)
    second = tune_hyperparameters(**kwargs)
    assert first[1] == second[1]
    assert first[2] == second[2]
    np.testing.assert_allclose(first[3].cv_results_["mean_test_score"], second[3].cv_results_["mean_test_score"])
