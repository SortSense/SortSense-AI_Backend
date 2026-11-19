"""Unit tests for SortSense Pydantic schemas and standard bin mapping."""

import pytest
from pydantic import ValidationError

from app.schemas import (
    BIN_INDEX_TO_CATEGORY,
    CATEGORY_TO_BIN_INDEX,
    CloudTelemetryPayload,
    HealthResponse,
    VerifyDropRequest,
    VisionPredictionResponse,
    WasteCategory,
)


def test_standard_bin_mapping_enum_values():
    """Verify exact standardized category lowercase strings."""
    assert WasteCategory.PLASTIC.value == "plastic"
    assert WasteCategory.PAPER.value == "paper"
    assert WasteCategory.METAL.value == "metal"
    assert WasteCategory.GENERAL.value == "general"
    assert WasteCategory.NONE.value == "none"


def test_bidirectional_bin_mapping():
    """Verify exact index contracts: 0=plastic, 1=paper, 2=metal, 3=general, -1=none."""
    assert BIN_INDEX_TO_CATEGORY[0] == WasteCategory.PLASTIC
    assert BIN_INDEX_TO_CATEGORY[1] == WasteCategory.PAPER
    assert BIN_INDEX_TO_CATEGORY[2] == WasteCategory.METAL
    assert BIN_INDEX_TO_CATEGORY[3] == WasteCategory.GENERAL
    assert BIN_INDEX_TO_CATEGORY[-1] == WasteCategory.NONE

    assert CATEGORY_TO_BIN_INDEX[WasteCategory.PLASTIC] == 0
    assert CATEGORY_TO_BIN_INDEX[WasteCategory.PAPER] == 1
    assert CATEGORY_TO_BIN_INDEX[WasteCategory.METAL] == 2
    assert CATEGORY_TO_BIN_INDEX[WasteCategory.GENERAL] == 3
    assert CATEGORY_TO_BIN_INDEX[WasteCategory.NONE] == -1


def test_vision_prediction_response_valid():
    """Verify VisionPredictionResponse serialization and validation."""
    resp = VisionPredictionResponse(
        category=WasteCategory.PLASTIC,
        bin_index=0,
        confidence=0.91,
    )
    assert resp.category == WasteCategory.PLASTIC
    assert resp.bin_index == 0
    assert resp.confidence == 0.91

    dumped = resp.model_dump()
    assert dumped == {
        "category": "plastic",
        "bin_index": 0,
        "confidence": 0.91,
    }


def test_vision_prediction_response_invalid_confidence():
    """Verify confidence must be between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        VisionPredictionResponse(
            category=WasteCategory.PLASTIC,
            bin_index=0,
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        VisionPredictionResponse(
            category=WasteCategory.PLASTIC,
            bin_index=0,
            confidence=-0.1,
        )


def test_verify_drop_request_valid():
    """Verify VerifyDropRequest handles valid station_id and bin_index."""
    req = VerifyDropRequest(station_id="ST-001", bin_index=0)
    assert req.station_id == "ST-001"
    assert req.bin_index == 0


@pytest.mark.parametrize("invalid_index", [-1, 4, 10])
def test_verify_drop_request_invalid_bin_index(invalid_index):
    """Verify VerifyDropRequest rejects invalid bin indices outside 0..3."""
    with pytest.raises(ValidationError):
        VerifyDropRequest(station_id="ST-001", bin_index=invalid_index)


def test_verify_drop_request_empty_station():
    """Verify station_id cannot be empty."""
    with pytest.raises(ValidationError):
        VerifyDropRequest(station_id="", bin_index=0)


def test_cloud_telemetry_payload_serialization():
    """Verify CloudTelemetryPayload matches exact JSON schema expected by Cloud Website."""
    payload = CloudTelemetryPayload(
        station_id="ST-001",
        waste_category="plastic",
        dropped_in_bin="plastic",
        confidence=0.91,
        is_correct_sort=True,
    )
    dumped = payload.model_dump()
    assert dumped == {
        "station_id": "ST-001",
        "waste_category": "plastic",
        "dropped_in_bin": "plastic",
        "confidence": 0.91,
        "is_correct_sort": True,
    }


def test_health_response_defaults():
    """Verify HealthResponse default values and schema."""
    resp = HealthResponse(model_backend="yolov26")
    assert resp.status == "healthy"
    assert resp.service == "SortSense-AI_Backend"
    assert resp.model_backend == "yolov26"
    assert resp.version == "1.0.0"
    assert resp.active_station_count == 0
