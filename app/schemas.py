"""SortSense AI Backend Pydantic Schemas & Standardized Enums.

Defines the exact JSON contracts for the ESP32-S3 WebSocket stream,
internal drop verification, and Cloud Website telemetry dispatch.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class WasteCategory(str, Enum):
    """Standardized waste categories matching the SortSense ecosystem."""

    PLASTIC = "plastic"
    PAPER = "paper"
    METAL = "metal"
    GENERAL = "general"
    NONE = "none"


# Strict bidirectional bin mapping:
# 0 -> plastic, 1 -> paper, 2 -> metal, 3 -> general, -1 -> none
BIN_INDEX_TO_CATEGORY: dict[int, WasteCategory] = {
    0: WasteCategory.PLASTIC,
    1: WasteCategory.PAPER,
    2: WasteCategory.METAL,
    3: WasteCategory.GENERAL,
    -1: WasteCategory.NONE,
}

CATEGORY_TO_BIN_INDEX: dict[WasteCategory, int] = {
    WasteCategory.PLASTIC: 0,
    WasteCategory.PAPER: 1,
    WasteCategory.METAL: 2,
    WasteCategory.GENERAL: 3,
    WasteCategory.NONE: -1,
}


class VisionPredictionResponse(BaseModel):
    """Payload sent to ESP32-S3 over WS /ws/vision."""

    category: WasteCategory = Field(
        ...,
        description="Classified waste category: plastic, paper, metal, general, or none",
    )
    bin_index: int = Field(
        ...,
        description="Index of target bin (0=plastic, 1=paper, 2=metal, 3=general, -1=none)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score between 0.0 and 1.0",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "category": "plastic",
                "bin_index": 0,
                "confidence": 0.91,
            }
        }
    }


class VerifyDropRequest(BaseModel):
    """Payload received from ESP32-S3 IR sensor via POST /internal/verify-drop."""

    station_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Identifier of the sorting station (e.g. ST-001)",
    )
    bin_index: int = Field(
        ...,
        ge=0,
        le=3,
        description="Bin index where the object dropped (0=plastic, 1=paper, 2=metal, 3=general)",
    )

    @field_validator("bin_index")
    @classmethod
    def validate_bin_index(cls, v: int) -> int:
        if v not in (0, 1, 2, 3):
            raise ValueError(f"bin_index must be one of 0, 1, 2, 3. Received {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "station_id": "ST-001",
                "bin_index": 0,
            }
        }
    }


class VerifyDropResponse(BaseModel):
    """Response returned to caller of POST /internal/verify-drop."""

    station_id: str
    waste_category: str
    dropped_in_bin: str
    confidence: float
    is_correct_sort: bool
    status: str = Field(
        default="verified",
        description="Verification execution status (e.g. verified, forwarded)",
    )
    cloud_forwarded: bool = Field(
        default=False,
        description="Whether telemetry was successfully dispatched to Cloud API",
    )


class CloudTelemetryPayload(BaseModel):
    """Payload dispatched to Cloud Website API: POST {CLOUD_API_URL}/api/events/log-waste."""

    station_id: str = Field(..., description="Unique sorting station ID")
    waste_category: str = Field(..., description="Predicted waste category")
    dropped_in_bin: str = Field(..., description="Actual bin user dropped item into")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Inference confidence score")
    is_correct_sort: bool = Field(..., description="True if dropped_in_bin == waste_category")

    model_config = {
        "json_schema_extra": {
            "example": {
                "station_id": "ST-001",
                "waste_category": "plastic",
                "dropped_in_bin": "plastic",
                "confidence": 0.91,
                "is_correct_sort": True,
            }
        }
    }


class FillLevels(BaseModel):
    """Compartment fill levels in percentages (0-100)."""

    plastic: float = Field(default=0.0, ge=0.0, le=100.0)
    paper: float = Field(default=0.0, ge=0.0, le=100.0)
    metal: float = Field(default=0.0, ge=0.0, le=100.0)
    general: float = Field(default=0.0, ge=0.0, le=100.0)


class FillLevelsTelemetryPayload(BaseModel):
    """Payload sent by ESP32-S3 ToF sensors to POST /api/telemetry/fill-levels."""

    station_id: str = Field(..., min_length=1, max_length=64, description="Station identifier")
    fill_levels: FillLevels

    model_config = {
        "json_schema_extra": {
            "example": {
                "station_id": "ST-001",
                "fill_levels": {
                    "plastic": 45.0,
                    "paper": 80.0,
                    "metal": 20.0,
                    "general": 95.0,
                },
            }
        }
    }


class HealthResponse(BaseModel):
    """Healthcheck response model."""

    status: str = "healthy"
    service: str = "SortSense-AI_Backend"
    version: str = "1.0.0"
    model_backend: str
    active_station_count: int = 0
