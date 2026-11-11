import json
import pytest
from app import app, load_ml_pipeline

@pytest.fixture
def client():
    app.config['TESTING'] = True
    load_ml_pipeline()
    with app.test_client() as client:
        yield client

@pytest.fixture
def valid_pair_payload():
    return {
        "a_age": 28, "b_age": 30,
        "a_education": 4, "b_education": 4,
        "a_location": "Urban", "b_location": "Urban",
        "a_career_field": "Tech", "b_career_field": "Tech",
        "a_career_ambition": 0.8, "b_career_ambition": 0.75,
        "a_openness": 0.7, "b_openness": 0.65,
        "a_extraversion": 0.6, "b_extraversion": 0.55,
        "a_agreeableness": 0.8, "b_agreeableness": 0.85,
        "a_conscientiousness": 0.75, "b_conscientiousness": 0.7,
        "a_chronotype": 0.5, "b_chronotype": 0.5,
        "a_spontaneity": 0.6, "b_spontaneity": 0.65,
        "a_love_language": "Quality Time", "b_love_language": "Quality Time",
        "a_emotional_expressiveness": 0.7, "b_emotional_expressiveness": 0.7
    }

def test_app_index_route(client):
    response = client.get('/')
    assert response.status_code == 200

def test_app_health_route(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_loaded" in data

def test_app_model_info_route(client):
    response = client.get('/api/model-info')
    assert response.status_code in [200, 500]

def test_app_predict_missing_fields_400(client):
    incomplete_payload = {"a_age": 25, "b_age": 26}
    response = client.post('/api/predict', json=incomplete_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Missing required field" in data["error"]

def test_app_predict_invalid_numeric_field_400(client, valid_pair_payload):
    invalid_payload = valid_pair_payload.copy()
    invalid_payload["a_age"] = "not_a_number"
    response = client.post('/api/predict', json=invalid_payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "must be numeric" in data["error"]

def test_app_predict_empty_payload_400(client):
    response = client.post('/api/predict', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False

def test_app_predict_valid_payload_success(client, valid_pair_payload):
    response = client.post('/api/predict', json=valid_pair_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "compatibility_score" in data
    assert "compatible" in data
    assert data["compatible"] in [0, 1]

def test_app_predict_response_contains_xai(client, valid_pair_payload):
    response = client.post('/api/predict', json=valid_pair_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "xai_breakdown" in data
    assert "strengths" in data["xai_breakdown"]
    assert "frictions" in data["xai_breakdown"]

def test_app_predict_compatibility_bounds(client, valid_pair_payload):
    response = client.post('/api/predict', json=valid_pair_payload)
    assert response.status_code == 200
    data = response.get_json()
    score = data["compatibility_score"]
    assert 0.0 <= score <= 100.0
    longevity = data["estimated_longevity_months"]
    assert 3 <= longevity <= 120

def test_app_predict_contains_archetypes(client, valid_pair_payload):
    response = client.post('/api/predict', json=valid_pair_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "archetype_a" in data
    assert "archetype_b" in data
    assert "archetype_affinity" in data
    assert "name" in data["archetype_a"]
    assert "affinity_pct" in data["archetype_affinity"]

def test_app_predict_normalizes_1_to_10_traits_seamlessly(client, valid_pair_payload):
    # Scale all traits to 1-10
    payload_10 = valid_pair_payload.copy()
    for k in ["a_career_ambition", "b_career_ambition", "a_openness", "b_openness", "a_extraversion", "b_extraversion"]:
        payload_10[k] = payload_10[k] * 10.0
    response = client.post('/api/predict', json=payload_10)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert 0.0 <= data["compatibility_score"] <= 100.0
