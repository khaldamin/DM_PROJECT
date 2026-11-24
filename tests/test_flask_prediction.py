"""
Unit tests for the Flask /api/predict endpoint and related endpoints.
Uses Flask test client — no server needs to be running.
"""
import json
import sys
import os

import pytest

# Ensure project root is on sys.path (pytest.ini already does this, but be explicit)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the Flask app object without triggering load_ml_pipeline at module level
from app import app as flask_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a Flask test client with TESTING mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


VALID_PAYLOAD = {
    "a_age": 28,
    "b_age": 25,
    "a_education": 3,
    "b_education": 3,
    "a_career_ambition": 7,
    "b_career_ambition": 6,
    "a_openness": 8,
    "b_openness": 7,
    "a_extraversion": 6,
    "b_extraversion": 5,
    "a_agreeableness": 7,
    "b_agreeableness": 8,
    "a_conscientiousness": 7,
    "b_conscientiousness": 6,
    "a_chronotype": 5,
    "b_chronotype": 6,
    "a_spontaneity": 6,
    "b_spontaneity": 7,
    "a_emotional_expressiveness": 7,
    "b_emotional_expressiveness": 8,
    "a_location": "Paris",
    "b_location": "Paris",
    "a_career_field": "Engineering",
    "b_career_field": "Science",
    "a_love_language": "Words of Affirmation",
    "b_love_language": "Words of Affirmation",
}


# ---------------------------------------------------------------------------
# Tests — /api/predict
# ---------------------------------------------------------------------------

class TestPredictEndpoint:

    def test_valid_payload_returns_200(self, client):
        """A fully-populated payload must yield HTTP 200."""
        response = client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert response.status_code == 200

    def test_valid_payload_response_structure(self, client):
        """Response JSON must contain all expected keys with correct types."""
        response = client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        body = response.get_json()

        assert body is not None, "Response body must be valid JSON"
        assert body.get("success") is True, "success must be True"

        expected_keys = [
            "compatibility_score",
            "compatible",
            "estimated_longevity_months",
            "big5_harmony_score",
            "avg_trait_diff",
            "personality_distance",
        ]
        for key in expected_keys:
            assert key in body, f"Key '{key}' missing from response"

    def test_compatibility_score_in_valid_range(self, client):
        """compatibility_score must be a number between 0 and 100."""
        response = client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        body = response.get_json()
        score = body["compatibility_score"]
        assert isinstance(score, (int, float)), "compatibility_score must be numeric"
        assert 0 <= score <= 100, f"compatibility_score out of range: {score}"

    def test_compatible_is_binary(self, client):
        """compatible field must be 0 or 1."""
        response = client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        body = response.get_json()
        assert body["compatible"] in (0, 1), (
            f"compatible must be 0 or 1, got {body['compatible']}"
        )

    def test_longevity_in_valid_range(self, client):
        """estimated_longevity_months must be between 3 and 120."""
        response = client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        body = response.get_json()
        longevity = body["estimated_longevity_months"]
        assert 3 <= longevity <= 120, f"longevity out of range: {longevity}"

    def test_invalid_payload_returns_400(self, client):
        """Sending an empty JSON object should return HTTP 400."""
        response = client.post(
            "/api/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_invalid_payload_error_message(self, client):
        """400 response must include a descriptive error field."""
        response = client.post(
            "/api/predict",
            data=json.dumps({}),
            content_type="application/json",
        )
        body = response.get_json()
        assert body is not None
        assert body.get("success") is False
        assert "error" in body

    def test_missing_required_field_returns_400(self, client):
        """Omitting a required numeric field must return HTTP 400."""
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "a_age"}
        response = client.post(
            "/api/predict",
            data=json.dumps(incomplete),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_missing_location_returns_400(self, client):
        """Omitting a_location must return HTTP 400."""
        incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "a_location"}
        response = client.post(
            "/api/predict",
            data=json.dumps(incomplete),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_non_json_body_returns_400(self, client):
        """Sending non-JSON body must return HTTP 400."""
        response = client.post(
            "/api/predict",
            data="this is not json",
            content_type="text/plain",
        )
        assert response.status_code == 400

    def test_different_locations_accepted(self, client):
        """Different cities for a and b must still produce a valid response."""
        payload = dict(VALID_PAYLOAD)
        payload["b_location"] = "Lyon"
        response = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body.get("success") is True

    def test_different_love_languages_accepted(self, client):
        """Mismatched love languages must still produce a valid response."""
        payload = dict(VALID_PAYLOAD)
        payload["b_love_language"] = "Acts of Service"
        response = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body.get("success") is True


# ---------------------------------------------------------------------------
# Tests — /api/health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """GET /api/health must return HTTP 200."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        """Health response must contain status and model_loaded keys."""
        response = client.get("/api/health")
        body = response.get_json()
        assert body is not None
        assert "status" in body
        assert "model_loaded" in body

    def test_health_status_value(self, client):
        """status field must be a recognisable string ('ok', 'healthy', …)."""
        response = client.get("/api/health")
        body = response.get_json()
        assert isinstance(body["status"], str)
        assert body["status"].lower() in ("ok", "healthy", "ready", "degraded")

    def test_health_model_loaded_is_bool(self, client):
        """model_loaded must be a boolean."""
        response = client.get("/api/health")
        body = response.get_json()
        assert isinstance(body["model_loaded"], bool)


# ---------------------------------------------------------------------------
# Tests — /api/model-info
# ---------------------------------------------------------------------------

class TestModelInfoEndpoint:

    def test_model_info_returns_200(self, client):
        """GET /api/model-info must return HTTP 200."""
        response = client.get("/api/model-info")
        assert response.status_code == 200

    def test_model_info_response_structure(self, client):
        """model-info must contain best_model and metrics keys."""
        response = client.get("/api/model-info")
        body = response.get_json()
        assert body is not None
        assert "best_model" in body
        assert "metrics" in body

