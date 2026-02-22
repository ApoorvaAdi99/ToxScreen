import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from api.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_rewrite_safe_selection():
    """Test that rewrite_safe chooses the best candidate."""
    mock_original_scores = {"labels": {"toxicity": 0.9}, "overall": 0.9}
    
    # Mock candidates returned by vLLM
    mock_candidates = ["safe rewrite 1", "toxic rewrite 2"]
    
    # Mock classifier scores for candidates
    # Results should be deterministic for selection logic test
    def mock_score_text(text):
        if "safe" in text:
            return {"labels": {"toxicity": 0.1}, "overall": 0.1}
        else:
            return {"labels": {"toxicity": 0.8}, "overall": 0.8}

    with patch("api.main.classifier.is_loaded", True):
        with patch("api.main.classifier.score_text", side_effect=[mock_original_scores, {"labels": {"toxicity": 0.1}, "overall": 0.1}, {"labels": {"toxicity": 0.8}, "overall": 0.8}]):
            with patch("api.rewrite.vllm_client.generate_n_candidates", new_callable=AsyncMock) as mock_gen:
                mock_gen.return_value = mock_candidates
                
                response = client.post("/rewrite_safe", json={"text": "toxic input"})
                
                assert response.status_code == 200
                data = response.json()
                assert data["rewrite"] == "safe rewrite 1"
                assert data["rewrite_scores"]["overall"] == 0.1
                assert data["original"]["overall"] == 0.9
                assert data["reduction_failed"] == False

@pytest.mark.asyncio
async def test_rewrite_safe_vllm_error():
    """Test handling of vLLM service errors."""
    with patch("api.main.classifier.is_loaded", True):
        with patch("api.main.classifier.score_text", return_value={"labels": {"toxicity": 0.9}, "overall": 0.9}):
            with patch("api.rewrite.vllm_client.generate_n_candidates", new_callable=AsyncMock) as mock_gen:
                mock_gen.side_effect = Exception("vLLM down")
                
                response = client.post("/rewrite_safe", json={"text": "toxic input"})
                assert response.status_code == 503
                assert "vLLM service error" in response.json()["detail"]
