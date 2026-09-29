"""
test_api.py
───────────
Integration and unit tests for the FastAPI predictive maintenance API service.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.api.app import app, get_alert_level

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "model_loaded" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    # Returns 200 if model checkpoint exists, or 503 if not loaded yet
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "n_features" in data


def test_alert_level_mapping():
    assert get_alert_level(0.10) == "NORMAL"
    assert get_alert_level(0.45) == "WARNING"
    assert get_alert_level(0.70) == "HIGH"
    assert get_alert_level(0.95) == "CRITICAL"


def test_predict_validation_error():
    # Sending invalid shape or empty data should return 422 Unprocessable Entity
    response = client.post("/predict", json={"sensor_window": [], "unit_id": "test-unit"})
    assert response.status_code == 422
