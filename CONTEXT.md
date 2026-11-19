# SortSense-AI_Backend — Repo Context & AI Instructions
> **Repo:** `SortSense-AI_Backend` `https://github.com/SortSense/SortSense-AI_Backend`
> **Usage:** Place this file in the repo root as `CONTEXT.md` (or `.agentrules`). It defines this repo's exact endpoints and payloads-
so it connects seamlessly with https://github.com/SortSense/SortSense-Hardware and https://github.com/SortSense/SortSense-Website.

## 1. Role in the SortSense System
This repository is the **Local LAN Vision/LLM Inference Server** (Dockerized Python FastAPI). Because Vercel cannot hold persistent WebSockets, this local server handles all low-latency communication with the ESP32-S3:
1. Accepts real-time binary JPEG frames from the ESP32-S3 over **`WS /ws/vision`**.
2. Classifies the waste item using our pluggable vision classifier (`app/detector.py` — supports Vision LLM / VLM or YOLOv8 / YOLOv26).
3. Immediately replies over the WebSocket with the predicted bin index so the ESP32 flashes the right LED.
4. Caches the latest prediction per `station_id`. When the ESP32 calls **`POST /internal/verify-drop`**, this server compares the dropped bin against the predicted category and forwards the verified event via HTTP `POST` to the Cloud Website API.

---

## 2. Tech Stack & Constraints
* **Resolution of the image:** 640 × 480 pixels
* **Language:** Python 3.11+ (helps satisfy the project's **>50% Python codebase** requirement).
* **Framework:** FastAPI + Uvicorn (WebSockets + Async HTTP).
* **HTTP Client:** `httpx` (async) for forwarding events to the Vercel Cloud API.
* **Deployment:** Local machine / laptop on the same LAN/hotspot as the ESP32, packaged with a `Dockerfile` and `.env` (`CLOUD_API_URL`, model/LLM API keys).
* **Modular Classifier Design:** Isolate model inference inside `app/detector.py` behind a clean interface `classify_frame(jpeg_bytes: bytes) -> dict` so we can swap between a Vision LLM (e.g., Gemini/OpenAI/Ollama VLM) and YOLOv26/YOLOv8 without touching any WebSocket or FastAPI routes.

---

## 3. Standardized Bin Mapping (Strict Enum)
Always map model outputs to these exact lowercase strings and indices:
* `0` → `"plastic"`
* `1` → `"paper"`
* `2` → `"metal"`
* `3` → `"general"`
* `-1` → `"none"` (no trash detected / low confidence)

---

## 4. API Contracts (Strict Handshake)

### A. Endpoints This Repo OWNS (Called by ESP32-S3)

#### 1. `WS /ws/vision`
* **Incoming (Binary):** Raw 640x480 JPEG image bytes from ESP32-S3.
* **Outgoing (Text JSON):**
```json
{
  "category": "plastic",
  "bin_index": 0,
  "confidence": 0.91
}
```
*(Throttles or debounces classification so we don't spam the Vision LLM API or flicker the LEDs on every frame).*

#### 2. `POST /internal/verify-drop`
* **Incoming JSON (from ESP32-S3 IR sensor):**
```json
{
  "station_id": "ST-001",
  "bin_index": 0
}
```
* **Behavior:**
1. Convert `bin_index` to `dropped_in_bin` (`0="plastic"`, `1="paper"`, `2="metal"`, `3="general"`).
2. Look up the most recent cached prediction (`waste_category`, `confidence`).
3. Set `is_correct_sort = (dropped_in_bin == waste_category)`.
4. Forward asynchronously via `POST` to the SortSense Cloud API below.

### B. Endpoint This Repo CALLS (Hosted on the Cloud Website)
#### `POST {CLOUD_API_URL}/api/events/log-waste`
* **Outgoing JSON:**
```json
{
  "station_id": "ST-001",
  "waste_category": "plastic",
  "dropped_in_bin": "plastic",
  "confidence": 0.91,
  "is_correct_sort": true
}
```

---

## 5. Recommended Repo Structure
```text
SortSense-AI_Backend/
├── CONTEXT.md
├── Dockerfile
├── requirements.txt
├── .env.example
└── app/
    ├── main.py        # FastAPI app, WS /ws/vision, POST /internal/verify-drop
    ├── detector.py    # Pluggable LLM/VLM or YOLO inference + class mapping
    ├── cloud_client.py # Async httpx client posting to Vercel Cloud API
    └── schemas.py     # Pydantic models matching the exact JSON contracts
```
