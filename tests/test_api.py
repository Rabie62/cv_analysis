"""Unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "en" in data["supported_languages"]


def test_api_analyze_validation_empty():
    response = client.post("/api/analyze/text", json={"cv_text": "", "job_description_text": ""})
    assert response.status_code == 400
