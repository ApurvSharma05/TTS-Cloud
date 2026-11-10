"""
test_api.py - Integration tests for the TTS-Cloud FastAPI endpoints.
Tests health checks, Pydantic input validation, and endpoint schemas.
"""

import pytest
from fastapi.testclient import TestClient
from api import app
from config import get_cors_config, get_max_upload_bytes

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


def test_ingest_rejects_oversized_upload_with_413(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.001")  # ~1 KB limit
    assert get_max_upload_bytes() == 1048

    # Small overshoot: caught by the endpoint's size check
    files = {"file": ("big.pdf", b"x" * 5_000, "application/pdf")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()

    # Large overshoot: rejected from Content-Length before the body is parsed
    files = {"file": ("huge.pdf", b"x" * 200_000, "application/pdf")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 413


def test_ingest_empty_file_returns_400():
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/api/v1/ingest", files=files)
    assert response.status_code == 400


def test_default_upload_limit_is_20mb(monkeypatch):
    monkeypatch.delenv("MAX_UPLOAD_MB", raising=False)
    assert get_max_upload_bytes() == 20 * 1024 * 1024


def test_cors_never_combines_wildcard_with_credentials(monkeypatch):
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
    assert get_cors_config() == (["*"], False)

    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://a.example, https://b.example")
    assert get_cors_config() == (["https://a.example", "https://b.example"], True)


def test_cors_allows_configured_origin_only():
    allowed = client.get("/api/v1/health", headers={"Origin": "http://localhost:8501"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-credentials" not in allowed.headers

    blocked = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers
