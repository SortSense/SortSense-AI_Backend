# ==============================================================================
# SortSense-AI_Backend — Production Dockerfile
# Multi-stage lightweight container for Local LAN Vision/LLM Inference Server
# ==============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt .

# Install Python packages into isolated wheels / prefix
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Final Runtime Stage
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install essential runtime shared libraries (libGL and glib for Pillow/OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Copy installed python libraries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY CONTEXT.md .
COPY explanation.md .
COPY .env.example .

# Create directory for weights and datasets with correct permissions
RUN mkdir -p /app/weights /app/dataset && chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_BACKEND=yolov26 \
    FRAME_DEBOUNCE_INTERVAL=0.15

EXPOSE 8000

# Container healthcheck targeting /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]
