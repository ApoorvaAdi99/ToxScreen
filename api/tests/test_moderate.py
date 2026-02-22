import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from api.main import app

client = TestClient(app)

def test_moderate_endpoint_success():
    mock_result = {
        "labels": {"toxicity": 0.9},
        "overall": 0.9
    }
    with patch("api.main.classifier.is_loaded", True):
        with patch("api.main.classifier.score_text", return_value=mock_result):
            response = client.post("/moderate", json={"text": "some toxic text"})
            assert response.status_code == 200
            assert response.json() == mock_result

def test_moderate_endpoint_not_loaded():
    with patch("api.main.classifier.is_loaded", False):
        response = client.post("/moderate", json={"text": "some text"})
        assert response.status_code == 503
        assert response.json()["detail"] == "Classifier not loaded"

def test_moderate_endpoint_empty_text():
    response = client.post("/moderate", json={"text": ""})
    # FastAPI/Pydantic validation might allow empty text unless min_length=1 is set
    # Let's see how our schema handles it.
    assert response.status_code == 200
