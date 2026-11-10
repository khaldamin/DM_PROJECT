import os
import numpy as np
import pytest
from src.models.base_model import BaseModel
from src.models.logistic_regression import CupidLogisticRegression
from src.models.decision_tree import CupidDecisionTree
from src.models.random_forest import CupidRandomForest
from src.models.xgboost_model import CupidXGBoost

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.binomial(1, 0.5, 100)
    return X, y

def test_base_model_abstract_instantiation_raises():
    with pytest.raises(TypeError):
        BaseModel("test")

def test_logistic_regression_fit_predict(dummy_data):
    X, y = dummy_data
    clf = CupidLogisticRegression(max_iter=100, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

def test_logistic_regression_predict_proba(dummy_data):
    X, y = dummy_data
    clf = CupidLogisticRegression(max_iter=100, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_decision_tree_fit_predict(dummy_data):
    X, y = dummy_data
    clf = CupidDecisionTree(max_depth=3, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)

def test_decision_tree_feature_importances(dummy_data):
    X, y = dummy_data
    clf = CupidDecisionTree(max_depth=3, random_state=42)
    clf.fit(X, y)
    importances = clf.get_feature_importances()
    assert len(importances) == X.shape[1]
    assert np.isclose(importances.sum(), 1.0)

def test_random_forest_fit_predict(dummy_data):
    X, y = dummy_data
    clf = CupidRandomForest(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)

def test_random_forest_feature_importances(dummy_data):
    X, y = dummy_data
    clf = CupidRandomForest(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    importances = clf.get_feature_importances()
    assert len(importances) == X.shape[1]
    assert np.isclose(importances.sum(), 1.0)

def test_xgboost_fit_predict(dummy_data):
    X, y = dummy_data
    clf = CupidXGBoost(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(y)

def test_xgboost_feature_importances(dummy_data):
    X, y = dummy_data
    clf = CupidXGBoost(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    importances = clf.get_feature_importances()
    assert len(importances) == X.shape[1]

def test_model_save_and_load(tmp_path, dummy_data):
    X, y = dummy_data
    clf = CupidLogisticRegression(max_iter=100, random_state=42)
    clf.fit(X, y)
    save_path = str(tmp_path / "test_model.pkl")
    clf.save(save_path)
    loaded = BaseModel.load(save_path)
    assert loaded.model_name == clf.model_name
    np.testing.assert_array_equal(clf.predict(X), loaded.predict(X))

