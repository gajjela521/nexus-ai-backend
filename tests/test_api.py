from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_scanner_match():
    payload = {
        "resume_text": "I am a software engineer with python and react skills.",
        "job_description": "Looking for a python engineer."
    }
    response = client.post("/api/v1/scanner/match", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "match_level" in data
