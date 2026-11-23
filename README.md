# SortSense-AI_Backend

> **Real-Time Edge Vision & AI Inference Server for SortSense Smart Waste Bin**

[![CI - SortSense AI Backend](https://github.com/SortSense/SortSense-AI_Backend/actions/workflows/ci.yml/badge.svg)](https://github.com/SortSense/SortSense-AI_Backend/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## Overview

`SortSense-AI_Backend` is the local high-performance LAN vision inference server for the **SortSense Smart Waste Management Ecosystem**. It bridges edge hardware with cloud analytics:

- **SortSense-Hardware**: [SortSense/SortSense-Hardware](https://github.com/SortSense/SortSense-Hardware) (ESP32-S3 Camera & IR break-beam sensors)
- **SortSense-Website**: [SortSense/SortSense-Website](https://github.com/SortSense/SortSense-Website) (Next.js Cloud Platform on Vercel)

For complete architectural details, sequence diagrams, and fine-tuning documentation, see [explanation.md](explanation.md), [FINETUNING.md](FINETUNING.md), and [CONTEXT.md](CONTEXT.md).

---

## Features

- **Real-time Edge Classification:** Receives binary JPEG frames via `WS /ws/vision`, applies 5-frame temporal smoothing + handheld center-zone focus, and returns predicted bin index.
- **Fine-Tuned YOLO26n + Pluggable Backends:** Ships with fine-tuned `weights/yolo26n.pt` (`0.736 mAP50` on 783 held-out test images) and supports swapping to Vision LLMs (Gemini / OpenAI / Ollama) via environment variables.
- **Strict Bin Mapping:** Standardized across hardware, AI, and cloud:
  - `0`: Plastic
  - `1`: Paper
  - `2`: Metal
  - `3`: General
  - `-1`: None
- **Drop Verification & Telemetry Forwarding:** `POST /internal/verify-drop` and `POST /api/telemetry/fill-levels` asynchronously forward verified drop events and ToF compartment fill levels to the cloud backend.
- **Full Docker & CI/CD Support:** Production multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow.

---

## Quick Start

### 1. Local Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install ultralytics    # Required to load trained weights/yolo26n.pt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Live Webcam Test
```bash
python scripts/test_webcam.py
```

### 3. Docker Setup
```bash
docker compose up --build -d
```

### 4. Run Tests
```bash
pytest -v --cov=app --cov=scripts
```

---

## License
MIT License.
