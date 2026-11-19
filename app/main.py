"""SortSense-AI_Backend — Main FastAPI Application.

Local LAN Vision & LLM Inference Server for ESP32-S3 Camera and IR Sensors.
Endpoints:
  - WS   /ws/vision              (Stream 640x480 binary JPEG, return predicted bin index)
  - POST /internal/verify-drop   (IR sensor drop event verification & cloud telemetry)
  - GET  /health                 (Service health and active status)
  - GET  /                       (API info)
"""

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    FastAPI,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware

from app.cloud_client import get_cloud_client
from app.detector import classify_frame, get_detector
from app.schemas import (
    BIN_INDEX_TO_CATEGORY,
    CloudTelemetryPayload,
    FillLevelsTelemetryPayload,
    HealthResponse,
    VerifyDropRequest,
    VerifyDropResponse,
    VisionPredictionResponse,
    WasteCategory,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SortSense-AI_Backend")

# Global in-memory cache for station predictions:
# station_id -> { "category": str, "bin_index": int, "confidence": float, "timestamp": float }
station_prediction_cache: dict[str, dict[str, Any]] = {}

# Frame debounce interval in seconds (prevents frame spamming & LED flicker)
FRAME_DEBOUNCE_INTERVAL = float(os.getenv("FRAME_DEBOUNCE_INTERVAL", "0.15"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for clean startup and resource release."""
    logger.info("SortSense-AI_Backend initializing...")
    # Pre-initialize detector and cloud client
    detector = get_detector()
    logger.info(f"Vision detector backend ready: {type(detector).__name__}")
    cloud_client = get_cloud_client()
    logger.info(f"Cloud client configured for target: {cloud_client.base_url}")

    yield

    logger.info("Shutting down SortSense-AI_Backend...")
    await cloud_client.close()
    logger.info("Resources released cleanly.")


app = FastAPI(
    title="SortSense-AI_Backend",
    description="Local LAN Vision & LLM Inference Server for SortSense Smart Waste Bin",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards or monitoring tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root_info() -> dict[str, Any]:
    """Root info endpoint."""
    return {
        "service": "SortSense-AI_Backend",
        "description": "Local LAN Vision/LLM Inference Server for ESP32-S3",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def healthcheck() -> HealthResponse:
    """Service health status and active cache metrics."""
    backend_name = os.getenv("MODEL_BACKEND", "yolov26")
    return HealthResponse(
        status="healthy",
        service="SortSense-AI_Backend",
        version="1.0.0",
        model_backend=backend_name,
        active_station_count=len(station_prediction_cache),
    )


@app.websocket("/ws/vision")
async def vision_websocket(
    websocket: WebSocket,
    station_id: str = Query(default="ST-001", description="Sorting station identifier"),
) -> None:
    """Accept real-time binary JPEG frames from ESP32-S3 over WebSocket.

    Replies immediately with predicted bin index so ESP32 flashes the correct LED:
    {
      "category": "plastic",
      "bin_index": 0,
      "confidence": 0.91
    }
    """
    await websocket.accept()
    logger.info(f"ESP32-S3 connected to /ws/vision for station: '{station_id}'")

    last_inference_time = 0.0
    last_response: dict[str, Any] | None = None

    try:
        while True:
            # Receive raw message (can be bytes or text ping)
            message = await websocket.receive()
            if message.get("bytes"):
                jpeg_bytes: bytes = message["bytes"]
                now = time.monotonic()

                # Debounce: if frames arrive faster than debounce interval, reuse last result
                if (
                    now - last_inference_time
                ) < FRAME_DEBOUNCE_INTERVAL and last_response is not None:
                    await websocket.send_json(last_response)
                    continue

                # Run classification via pluggable detector interface
                result = classify_frame(jpeg_bytes)
                last_inference_time = now

                # Construct validated response
                prediction = VisionPredictionResponse(
                    category=WasteCategory(result["category"]),
                    bin_index=int(result["bin_index"]),
                    confidence=float(result["confidence"]),
                )

                # Update in-memory station cache for subsequent drop verification
                station_prediction_cache[station_id] = {
                    "category": prediction.category.value,
                    "bin_index": prediction.bin_index,
                    "confidence": prediction.confidence,
                    "timestamp": now,
                }

                last_response = prediction.model_dump()
                await websocket.send_json(last_response)

            elif message.get("text"):
                text_payload = message["text"].strip().lower()
                if text_payload == "ping":
                    await websocket.send_text("pong")
                else:
                    logger.debug(f"Received text message on /ws/vision: {text_payload}")

    except (WebSocketDisconnect, RuntimeError) as exc:
        if "disconnect" in str(exc).lower():
            logger.info(f"ESP32-S3 disconnected from /ws/vision for station: '{station_id}'")
        else:
            logger.error(f"Runtime error in vision WebSocket stream: {exc}", exc_info=True)
    except Exception as exc:
        logger.error(f"Error in vision WebSocket stream: {exc}", exc_info=True)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass


@app.post(
    "/internal/verify-drop",
    response_model=VerifyDropResponse,
    status_code=status.HTTP_200_OK,
    tags=["Verification"],
)
async def verify_drop(
    request: VerifyDropRequest,
    background_tasks: BackgroundTasks,
) -> VerifyDropResponse:
    """Verify drop event from ESP32-S3 IR sensor and forward telemetry to Cloud API.

    1. Convert bin_index to dropped_in_bin (0=plastic, 1=paper, 2=metal, 3=general).
    2. Look up most recent cached prediction (waste_category, confidence).
    3. Set is_correct_sort = (dropped_in_bin == waste_category).
    4. Forward asynchronously via POST to SortSense Cloud API.
    """
    station_id = request.station_id
    bin_index = request.bin_index

    # 1. Convert bin_index to dropped_in_bin
    dropped_category = BIN_INDEX_TO_CATEGORY.get(bin_index, WasteCategory.GENERAL)
    dropped_in_bin = dropped_category.value

    # 2. Look up most recent cached prediction
    cached = station_prediction_cache.get(station_id)
    if cached:
        waste_category = cached.get("category", WasteCategory.NONE.value)
        confidence = float(cached.get("confidence", 0.0))
    else:
        logger.warning(
            f"No prior vision prediction cached for station '{station_id}'. Defaulting to 'none'."
        )
        waste_category = WasteCategory.NONE.value
        confidence = 0.0

    # 3. Determine if sort is correct
    is_correct_sort = (dropped_in_bin == waste_category) and (
        waste_category != WasteCategory.NONE.value
    )

    # 4. Prepare cloud telemetry payload
    telemetry_payload = CloudTelemetryPayload(
        station_id=station_id,
        waste_category=waste_category,
        dropped_in_bin=dropped_in_bin,
        confidence=confidence,
        is_correct_sort=is_correct_sort,
    )

    # Dispatch to cloud client
    cloud_client = get_cloud_client()
    # Schedule async delivery in background so ESP32-S3 receives verification immediately
    background_tasks.add_task(cloud_client.post_waste_event, telemetry_payload)

    logger.info(
        f"Drop verified for {station_id}: dropped={dropped_in_bin}, predicted={waste_category}, correct={is_correct_sort}"
    )

    return VerifyDropResponse(
        station_id=station_id,
        waste_category=waste_category,
        dropped_in_bin=dropped_in_bin,
        confidence=confidence,
        is_correct_sort=is_correct_sort,
        status="verified",
        cloud_forwarded=True,
    )


@app.post(
    "/api/telemetry/fill-levels",
    status_code=status.HTTP_200_OK,
    tags=["Telemetry"],
)
async def ingest_fill_levels(
    payload: FillLevelsTelemetryPayload,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Receive ToF compartment fill levels from ESP32-S3 and forward to Cloud Website API."""
    station_id = payload.station_id
    levels = payload.fill_levels

    # Rich formatted logging in AI server console
    lines = [f"📊 [TELEMETRY] Station: {station_id} Compartment Fill Levels:"]
    for category in ("plastic", "paper", "metal", "general"):
        pct = int(getattr(levels, category, 0.0))
        filled = min(10, max(0, int(round(pct / 10.0))))
        bar = "#" * filled + "-" * (10 - filled)
        lines.append(f"    • {category.capitalize():<8}: [{bar}] {pct:3d}%")
    logger.info("\n" + "\n".join(lines))

    # Forward to Cloud Website API in background
    cloud_client = get_cloud_client()
    background_tasks.add_task(cloud_client.post_fill_levels, payload)

    return {
        "status": "ok",
        "station_id": station_id,
        "cloud_forwarded": True,
    }


def get_cache() -> dict[str, dict[str, Any]]:
    """Helper to inspect station prediction cache (for testing)."""
    return station_prediction_cache


def clear_cache() -> None:
    """Helper to reset station prediction cache (for testing)."""
    station_prediction_cache.clear()
