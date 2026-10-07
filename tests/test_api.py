"""
test_api.py - Integration tests for the TTS-Cloud FastAPI endpoints.
Tests health checks, Pydantic input validation, and endpoint schemas.
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "TTS-Cloud" in data["service"]
    assert "groq_model" in data
    assert "uptime_seconds" in data


def test_query_validation_error_on_short_question():
    # Questions shorter than 3 characters should be rejected with 422 Unprocessable Entity
    response = client.post("/api/v1/query", json={"question": "a"})
    assert response.status_code == 422


def test_query_validation_error_on_invalid_role():
    # Chat message roles must be user/assistant/system
    payload = {
        "question": "What is normal form?",
        "chat_history": [{"role": "superadmin", "content": "hello"}]
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 422


def test_ingest_rejects_non_pdf_file():
    # Uploading a .txt file should return 400 Bad Request
    files = {"file": ("notes.txt", b"plain text notes", "text/plain")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 400
    assert "Only PDF" in response.json()["detail"]


def test_openapi_docs_accessible():
    response = client.get("/docs")
    assert response.status_code == 200
