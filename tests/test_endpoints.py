"""Integration tests for FastAPI REST endpoints and drop verification logic."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.cloud_client import get_cloud_client
from app.main import app, clear_cache, station_prediction_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global cache before each test."""
    clear_cache()
    yield
    clear_cache()


def test_root_endpoint():
    """Verify GET / returns service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "SortSense-AI_Backend"
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    """Verify GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_backend" in data
    assert data["active_station_count"] == 0


def test_verify_drop_correct_sort_plastic():
    """Verify correct sort when dropped bin matches cached vision prediction."""
    # Seed cache with vision prediction for ST-001: plastic (bin 0)
    station_prediction_cache["ST-001"] = {
        "category": "plastic",
        "bin_index": 0,
        "confidence": 0.95,
        "timestamp": 123456.0,
    }

    with patch.object(get_cloud_client(), "post_waste_event", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = True

        payload = {"station_id": "ST-001", "bin_index": 0}
        response = client.post("/internal/verify-drop", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["station_id"] == "ST-001"
        assert data["waste_category"] == "plastic"
        assert data["dropped_in_bin"] == "plastic"
        assert data["confidence"] == 0.95
        assert data["is_correct_sort"] is True
        assert data["status"] == "verified"

        # Verify background task dispatched telemetry
        mock_post.assert_called_once()
        sent_telemetry = mock_post.call_args[0][0]
        assert sent_telemetry.station_id == "ST-001"
        assert sent_telemetry.waste_category == "plastic"
        assert sent_telemetry.dropped_in_bin == "plastic"
        assert sent_telemetry.is_correct_sort is True


def test_verify_drop_incorrect_sort():
    """Verify incorrect sort when user drops plastic item into paper bin (bin 1)."""
    station_prediction_cache["ST-001"] = {
        "category": "plastic",
        "bin_index": 0,
        "confidence": 0.92,
        "timestamp": 123456.0,
    }

    with patch.object(get_cloud_client(), "post_waste_event", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = True

        # Dropped in paper bin (1) instead of plastic (0)
        payload = {"station_id": "ST-001", "bin_index": 1}
        response = client.post("/internal/verify-drop", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["station_id"] == "ST-001"
        assert data["waste_category"] == "plastic"
        assert data["dropped_in_bin"] == "paper"
        assert data["is_correct_sort"] is False

        mock_post.assert_called_once()
        sent_telemetry = mock_post.call_args[0][0]
        assert sent_telemetry.is_correct_sort is False


def test_verify_drop_no_cached_prediction():
    """Verify behaviour when ESP32 drops without any prior camera frames."""
    with patch.object(get_cloud_client(), "post_waste_event", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = True

        payload = {"station_id": "ST-UNKNOWN", "bin_index": 2}
        response = client.post("/internal/verify-drop", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["station_id"] == "ST-UNKNOWN"
        assert data["waste_category"] == "none"
        assert data["dropped_in_bin"] == "metal"
        assert data["confidence"] == 0.0
        assert data["is_correct_sort"] is False


@pytest.mark.parametrize("invalid_bin", [-1, 4, 99])
def test_verify_drop_validation_error_invalid_bin(invalid_bin):
    """Verify 422 Unprocessable Entity when bin_index is outside 0..3."""
    response = client.post(
        "/internal/verify-drop",
        json={"station_id": "ST-001", "bin_index": invalid_bin},
    )
    assert response.status_code == 422


def test_verify_drop_validation_error_empty_station():
    """Verify 422 Unprocessable Entity when station_id is empty."""
    response = client.post(
        "/internal/verify-drop",
        json={"station_id": "", "bin_index": 0},
    )
    assert response.status_code == 422


def test_ingest_fill_levels_success():
    """Verify POST /api/telemetry/fill-levels accepts and forwards telemetry."""
    with patch.object(get_cloud_client(), "post_fill_levels", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = True

        payload = {
            "station_id": "ST-001",
            "fill_levels": {
                "plastic": 45.0,
                "paper": 80.0,
                "metal": 20.0,
                "general": 95.0,
            },
        }
        response = client.post("/api/telemetry/fill-levels", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["station_id"] == "ST-001"
        assert data["cloud_forwarded"] is True


def test_ingest_fill_levels_validation_error():
    """Verify POST /api/telemetry/fill-levels returns 422 on invalid payload."""
    response = client.post(
        "/api/telemetry/fill-levels",
        json={"station_id": ""},
    )
    assert response.status_code == 422
