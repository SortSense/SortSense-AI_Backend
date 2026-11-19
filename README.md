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

For complete architectural details, sequence diagrams, and contract specifics, see [explanation.md](explanation.md) and [CONTEXT.md](CONTEXT.md).

---

## Features

- **Sub-25ms Real-time Classification:** Receives binary 640×480 JPEG frames via `WS /ws/vision` and returns predicted bin index.
- **Pluggable Model Architecture:** Swap seamlessly between **YOLOv26**, YOLOv8, and Vision LLMs (Gemini / OpenAI / Ollama) via environment variables.
- **Strict Bin Mapping:** Standardized across hardware, AI, and cloud:
  - `0`: Plastic
  - `1`: Paper
  - `2`: Metal
  - `3`: General
  - `-1`: None
- **Drop Verification & Cloud Forwarding:** `POST /internal/verify-drop` checks physical IR drop against vision prediction and asynchronously dispatches verified telemetry to the cloud backend.
- **Full Docker & CI/CD Support:** Production multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow.

---

## Quick Start

### 1. Local Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Docker Setup
```bash
docker compose up --build -d
```

### 3. Run Tests
```bash
pytest -v --cov=app --cov=scripts
```

---

## License
MIT License.
