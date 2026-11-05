"""Integration and Unit Tests for FastAPI Misconception Diagnostic Service."""
import base64
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api.service import app


@pytest.fixture
def client():
    return TestClient(app)


def test_healthcheck_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "gpu_telemetry" in data
    assert "cuda_available" in data["gpu_telemetry"]


def test_taxonomy_endpoint(client):
    response = client.get("/api/v1/taxonomy")
    assert response.status_code == 200
    data = response.json()
    assert data["total_codes"] >= 76
    assert len(data["categories"]) > 0
    assert len(data["misconceptions"]) >= 76


def test_diagnose_endpoint_success(client):
    # Create synthetic mock math handwriting image
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "trace_id": "test_trace_api_01",
        "grade": 7,
        "topic_name": "Operasi Bilangan Bulat",
        "question_text": "-3 - (-5) = ...",
        "correct_answer": "2",
        "image_base64": b64_str
    }

    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["trace_id"] == "test_trace_api_01"
    assert "clean_work" in res_data
    assert "transcribed_steps" in res_data
    assert len(res_data["top_3_predictions"]) == 3

    # Verify exactly 3 distinct codes
    codes = [p["root_code"] for p in res_data["top_3_predictions"]]
    assert len(set(codes)) == 3

    for pred in res_data["top_3_predictions"]:
        assert "root_code" in pred
        assert "name" in pred
        assert "score" in pred
    assert len(res_data["pedagogical_feedback"]) > 0


def test_diagnose_optional_fields(client):
    # Test without trace_id and without correct_answer
    img = Image.new("RGB", (50, 50), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "grade": 6,
        "topic_name": "Pecahan",
        "question_text": "1/2 + 1/4 = ?",
        "image_base64": b64_str
    }

    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["trace_id"].startswith("trace_")
    assert len(res_data["top_3_predictions"]) == 3


def test_diagnose_invalid_base64(client):
    payload = {
        "trace_id": "test_invalid_b64",
        "grade": 7,
        "topic_name": "Operasi Bilangan Bulat",
        "question_text": "-3 - (-5) = ...",
        "correct_answer": "2",
        "image_base64": "NOT_A_VALID_BASE64_STRING!@#$"
    }

    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 422
