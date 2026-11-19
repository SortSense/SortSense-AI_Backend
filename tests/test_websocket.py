"""Integration tests for WebSocket /ws/vision frame streaming."""

import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.detector import MockDetector, set_detector
from app.main import app, clear_cache, station_prediction_cache
from app.schemas import WasteCategory

client = TestClient(app)


def create_jpeg_bytes(width: int = 640, height: int = 480, color: tuple = (255, 0, 0)) -> bytes:
    """Create JPEG byte stream."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_detector():
    """Ensure clean detector and cache."""
    clear_cache()
    mock_det = MockDetector(category=WasteCategory.PLASTIC, confidence=0.94)
    set_detector(mock_det)
    yield
    clear_cache()


def test_websocket_ping_pong():
    """Verify WebSocket connection handles text ping messages."""
    with client.websocket_connect("/ws/vision?station_id=ST-TEST") as websocket:
        websocket.send_text("ping")
        data = websocket.receive_text()
        assert data == "pong"


def test_websocket_vision_stream_binary_jpeg():
    """Verify WebSocket accepts binary 640x480 JPEG and replies with predicted bin index."""
    jpeg_bytes = create_jpeg_bytes()

    with client.websocket_connect("/ws/vision?station_id=ST-001") as websocket:
        websocket.send_bytes(jpeg_bytes)
        response_json = websocket.receive_json()

        assert response_json["category"] == "plastic"
        assert response_json["bin_index"] == 0
        assert response_json["confidence"] == 0.94

        # Verify cached prediction for station ST-001
        assert "ST-001" in station_prediction_cache
        cached = station_prediction_cache["ST-001"]
        assert cached["category"] == "plastic"
        assert cached["bin_index"] == 0
        assert cached["confidence"] == 0.94


def test_websocket_vision_model_swap_in_stream():
    """Verify swapping detector reflects immediately in WebSocket stream without route restart."""
    with client.websocket_connect("/ws/vision?station_id=ST-002") as websocket:
        # First frame: plastic
        websocket.send_bytes(create_jpeg_bytes())
        r1 = websocket.receive_json()
        assert r1["category"] == "plastic"
        assert r1["bin_index"] == 0

        # Swap detector to Metal
        time.sleep(0.2)  # exceed debounce interval
        set_detector(MockDetector(category=WasteCategory.METAL, confidence=0.89))

        websocket.send_bytes(create_jpeg_bytes())
        r2 = websocket.receive_json()
        assert r2["category"] == "metal"
        assert r2["bin_index"] == 2
        assert r2["confidence"] == 0.89


def test_websocket_vision_debounce():
    """Verify rapid consecutive frames within debounce interval reuse last response."""
    jpeg_bytes = create_jpeg_bytes()

    with client.websocket_connect("/ws/vision?station_id=ST-DEBOUNCE") as websocket:
        websocket.send_bytes(jpeg_bytes)
        resp1 = websocket.receive_json()

        # Immediately send second frame without delay (< 0.15s)
        websocket.send_bytes(jpeg_bytes)
        resp2 = websocket.receive_json()

        assert resp1 == resp2
