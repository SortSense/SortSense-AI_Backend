"""SortSense Vision Classifier & Model Inference Module.

Provides a pluggable detector interface (`classify_frame`) supporting:
1. YOLOv26 (Ultra-fast edge AI object detection optimized for recyclable waste)
2. YOLOv8 fallback
3. Vision LLM / VLM (Gemini / OpenAI / Ollama multimodal inference)
4. Mock / Heuristic Detector (for tests, dry-runs, and resource-constrained environments)

All outputs strictly conform to the SortSense Bin Mapping:
  0  -> plastic
  1  -> paper
  2  -> metal
  3  -> general
 -1  -> none (no trash detected / low confidence)
"""

import abc
import base64
import io
import json
import logging
import os
from collections import deque
from typing import Any

import numpy as np
from PIL import Image

from app.schemas import (
    CATEGORY_TO_BIN_INDEX,
    WasteCategory,
)

logger = logging.getLogger(__name__)

# Standard target camera resolution for SortSense ESP32-S3
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480

# Class name to standardized SortSense category mapping
LABEL_TO_CATEGORY_MAP: dict[str, WasteCategory] = {
    # Plastic
    "bottle": WasteCategory.PLASTIC,
    "plastic": WasteCategory.PLASTIC,
    "plastic_bottle": WasteCategory.PLASTIC,
    "plastic_bag": WasteCategory.PLASTIC,
    "plastic_cup": WasteCategory.PLASTIC,
    "plastic_container": WasteCategory.PLASTIC,
    "wrapper": WasteCategory.PLASTIC,
    # Paper
    "paper": WasteCategory.PAPER,
    "cardboard": WasteCategory.PAPER,
    "box": WasteCategory.PAPER,
    "newspaper": WasteCategory.PAPER,
    "magazine": WasteCategory.PAPER,
    "carton": WasteCategory.PAPER,
    "paper_cup": WasteCategory.PAPER,
    # Metal
    "metal": WasteCategory.METAL,
    "can": WasteCategory.METAL,
    "aluminum_can": WasteCategory.METAL,
    "tin": WasteCategory.METAL,
    "tin_can": WasteCategory.METAL,
    "foil": WasteCategory.METAL,
    # General / Other
    "general": WasteCategory.GENERAL,
    "trash": WasteCategory.GENERAL,
    "food": WasteCategory.GENERAL,
    "organic": WasteCategory.GENERAL,
    "apple": WasteCategory.GENERAL,
    "banana": WasteCategory.GENERAL,
    "waste": WasteCategory.GENERAL,
    "battery": WasteCategory.GENERAL,
    # None
    "none": WasteCategory.NONE,
    "background": WasteCategory.NONE,
    "empty": WasteCategory.NONE,
}


def validate_jpeg(
    jpeg_bytes: bytes,
) -> tuple[bool, str | None, tuple[int, int] | None]:
    """Validate that incoming payload has JPEG header and parse its dimensions.

    Returns:
        (is_valid, error_message, (width, height))
    """
    if not jpeg_bytes or len(jpeg_bytes) < 4:
        return False, "Payload is empty or too short for a JPEG", None

    # JPEG standard SOI marker: 0xFF 0xD8
    if not (jpeg_bytes[0] == 0xFF and jpeg_bytes[1] == 0xD8):
        return False, "Invalid JPEG magic bytes (missing SOI marker)", None

    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as img:
            size = img.size  # (width, height)
            return True, None, size
    except Exception as exc:
        return False, f"Failed to parse JPEG image: {exc}", None


class BaseDetector(abc.ABC):
    """Abstract base class for all vision classifiers."""

    @abc.abstractmethod
    def classify_frame(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """Classify binary JPEG frame and return dict with category, bin_index, confidence."""
        raise NotImplementedError


class YOLOv26Detector(BaseDetector):
    """YOLOv26 Edge Classifier.

    YOLOv26 is chosen as the primary edge AI model for SortSense waste sorting:
    - Tailored for high-FPS edge inference (<25ms on CPU / edge NPU)
    - Precision anchor-free head with low parameter count
    - Optimized for recyclable waste classification (plastic, paper, metal, general)
    """

    def __init__(
        self,
        weights_path: str | None = None,
        confidence_threshold: float = 0.50,
        focus_center_roi: bool | None = None,
        min_box_area_ratio: float | None = None,
        max_box_area_ratio: float | None = None,
    ) -> None:
        self.weights_path = weights_path or os.getenv("MODEL_WEIGHTS_PATH", "weights/yolo26n.pt")
        self.confidence_threshold = float(
            os.getenv("CONFIDENCE_THRESHOLD", str(confidence_threshold))
        )
        if focus_center_roi is not None:
            self.focus_center_roi = focus_center_roi
        else:
            self.focus_center_roi = os.getenv("FOCUS_CENTER_ROI", "true").lower() in (
                "1",
                "true",
                "yes",
            )
        self.min_box_area_ratio = (
            min_box_area_ratio
            if min_box_area_ratio is not None
            else float(os.getenv("MIN_BOX_AREA_RATIO", "0.015"))
        )
        self.max_box_area_ratio = (
            max_box_area_ratio
            if max_box_area_ratio is not None
            else float(os.getenv("MAX_BOX_AREA_RATIO", "0.85"))
        )
        # Center scan zone bounds (normalized x1, y1, x2, y2) for handheld foreground focus
        self.roi_bounds: tuple[float, float, float, float] = (0.18, 0.18, 0.82, 0.88)
        self.model = None
        self.last_detections: list[dict[str, Any]] = []
        self.last_primary_detection: dict[str, Any] | None = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Attempt to load real YOLO model if ultralytics and weights exist; fallback gracefully."""
        if os.path.exists(self.weights_path):
            try:
                from ultralytics import YOLO

                self.model = YOLO(self.weights_path)
                logger.info(f"Loaded YOLO model weights from {self.weights_path}")
                return
            except Exception as exc:
                logger.warning(
                    f"Could not initialize ultralytics YOLO from {self.weights_path}: {exc}. "
                    "Falling back to built-in edge feature classifier."
                )
        else:
            logger.info(
                f"Weights file not found at '{self.weights_path}'. "
                "Running YOLOv26 edge simulation classifier (ready for inference/tests)."
            )

    def _score_detection(
        self,
        conf: float,
        xyxy: list[float] | None,
        img_size: tuple[int, int] | None,
    ) -> tuple[bool, float]:
        """Evaluate whether a detection is a valid foreground handheld object and score it.

        Returns:
            (is_in_focus, priority_score)
        """
        if not self.focus_center_roi or xyxy is None or not img_size:
            return True, conf

        img_w, img_h = img_size
        if img_w <= 0 or img_h <= 0:
            return True, conf

        x1, y1, x2, y2 = xyxy
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area_ratio = (box_w * box_h) / float(img_w * img_h)

        # Reject tiny distant background specks or full-frame wall/shirt false positives
        if area_ratio < self.min_box_area_ratio or area_ratio > self.max_box_area_ratio:
            return False, 0.0

        # Check whether the detection center sits inside the center scan zone
        cx = ((x1 + x2) / 2.0) / float(img_w)
        cy = ((y1 + y2) / 2.0) / float(img_h)
        rx1, ry1, rx2, ry2 = self.roi_bounds
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            return False, 0.0

        # Prioritize large, centered foreground items over peripheral clutter
        dist_from_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
        center_weight = max(0.0, 1.0 - dist_from_center / 0.5)
        size_weight = min(1.0, area_ratio / 0.20)
        priority = conf * (0.65 + 0.35 * center_weight) * (0.75 + 0.25 * size_weight)
        return True, priority

    def classify_frame(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """Classify a 640x480 JPEG frame using YOLOv26."""
        self.last_detections = []
        self.last_primary_detection = None
        is_valid, err_msg, dims = validate_jpeg(jpeg_bytes)
        if not is_valid:
            logger.warning(f"Invalid frame received: {err_msg}")
            return {
                "category": WasteCategory.NONE.value,
                "bin_index": -1,
                "confidence": 0.0,
            }

        # If real YOLO model is loaded, run forward pass
        if self.model is not None:
            try:
                with Image.open(io.BytesIO(jpeg_bytes)) as img:
                    img_size = img.size  # (width, height)
                    results = self.model(img, verbose=False)
                    best_low_conf = 0.0
                    best_primary: dict[str, Any] | None = None
                    best_priority = -1.0

                    for r in results:
                        if hasattr(r, "boxes") and len(r.boxes) > 0:
                            for b in r.boxes:
                                b_conf = float(b.conf[0])
                                b_cls = int(b.cls[0])
                                b_raw = r.names.get(b_cls, "unknown").lower()
                                b_cat = LABEL_TO_CATEGORY_MAP.get(b_raw, WasteCategory.GENERAL)
                                xyxy = (
                                    [float(v) for v in b.xyxy[0].tolist()]
                                    if hasattr(b, "xyxy") and len(b.xyxy) > 0
                                    else None
                                )
                                in_focus, priority = self._score_detection(b_conf, xyxy, img_size)
                                det_info = {
                                    "xyxy": xyxy,
                                    "raw_label": b_raw,
                                    "category": b_cat.value,
                                    "bin_index": CATEGORY_TO_BIN_INDEX[b_cat],
                                    "confidence": round(b_conf, 4),
                                    "above_threshold": b_conf >= self.confidence_threshold,
                                    "in_focus": in_focus,
                                    "priority": round(priority, 4),
                                }
                                self.last_detections.append(det_info)

                                if in_focus:
                                    if (
                                        b_conf >= self.confidence_threshold
                                        and priority > best_priority
                                    ):
                                        best_priority = priority
                                        best_primary = det_info
                                    elif b_conf < self.confidence_threshold:
                                        best_low_conf = max(best_low_conf, b_conf)

                    if best_primary is not None:
                        self.last_primary_detection = best_primary
                        return {
                            "category": best_primary["category"],
                            "bin_index": best_primary["bin_index"],
                            "confidence": best_primary["confidence"],
                        }

                    # Loaded model found no focused waste above threshold: do NOT
                    # fall through to the color heuristic, which would misclassify blank walls.
                    return {
                        "category": WasteCategory.NONE.value,
                        "bin_index": -1,
                        "confidence": round(best_low_conf, 4),
                    }
            except Exception as exc:
                logger.error(f"YOLO inference error: {exc}")

        # Edge feature heuristic simulation (used when weights file is not loaded)
        return self._heuristic_edge_classify(jpeg_bytes)

    def _heuristic_edge_classify(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """Edge heuristic feature extraction based on color temperature and saturation."""
        try:
            with Image.open(io.BytesIO(jpeg_bytes)).convert("RGB") as img:
                # Fast downsample for lightweight classification
                small = img.resize((32, 32))
                arr = np.array(small, dtype=float)
                # arr shape: (32, 32, 3)
                avg_r = float(np.mean(arr[:, :, 0]))
                avg_g = float(np.mean(arr[:, :, 1]))
                avg_b = float(np.mean(arr[:, :, 2]))
                brightness = (avg_r + avg_g + avg_b) / 3.0

                # Detect dark / empty frame
                if brightness < 15.0:
                    return {
                        "category": WasteCategory.NONE.value,
                        "bin_index": -1,
                        "confidence": 0.10,
                    }

                # Color dominance heuristic for edge demonstration & unit testing:
                # High Blue -> Plastic (e.g. blue bottle)
                # High Brightness / Low Saturation -> Paper (white/cardboard sheet)
                # High Metallic Reflection (gray/silver balanced) -> Metal
                # Warm Brown/Green/Yellow -> General
                diff_rg = abs(avg_r - avg_g)
                diff_gb = abs(avg_g - avg_b)
                diff_br = abs(avg_b - avg_r)
                is_gray = (diff_rg < 10) and (diff_gb < 10) and (diff_br < 10)

                if avg_b > avg_r * 1.25 and avg_b > avg_g * 1.15:
                    cat = WasteCategory.PLASTIC
                    conf = 0.91
                elif is_gray and brightness > 140:
                    cat = WasteCategory.METAL
                    conf = 0.88
                elif brightness > 180 and not is_gray:
                    cat = WasteCategory.PAPER
                    conf = 0.85
                elif avg_r > avg_b * 1.2 and avg_r > 80:
                    cat = WasteCategory.GENERAL
                    conf = 0.82
                else:
                    cat = WasteCategory.PLASTIC
                    conf = 0.78

                return {
                    "category": cat.value,
                    "bin_index": CATEGORY_TO_BIN_INDEX[cat],
                    "confidence": conf,
                }
        except Exception as exc:
            logger.warning(f"Heuristic extraction failure: {exc}")
            return {
                "category": WasteCategory.NONE.value,
                "bin_index": -1,
                "confidence": 0.0,
            }


class VisionLLMDetector(BaseDetector):
    """Vision Large Language Model (VLM) Detector.

    Supports multimodal inference via Gemini, OpenAI, or local Ollama.
    """

    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.getenv("VLM_PROVIDER", "gemini")).lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def classify_frame(self, jpeg_bytes: bytes) -> dict[str, Any]:
        """Classify frame using Vision LLM."""
        is_valid, err_msg, _ = validate_jpeg(jpeg_bytes)
        if not is_valid:
            return {
                "category": WasteCategory.NONE.value,
                "bin_index": -1,
                "confidence": 0.0,
            }

        b64_image = base64.b64encode(jpeg_bytes).decode("utf-8")
        prompt = (
            "Analyze this waste item for automated sorting. Respond ONLY with a valid JSON object: "
            '{"category": "plastic"|"paper"|"metal"|"general"|"none", "confidence": float between 0.0 and 1.0}.'
        )

        try:
            if self.provider == "gemini" and self.gemini_key:
                return self._call_gemini_vision(b64_image, prompt)
            elif self.provider == "openai" and self.openai_key:
                return self._call_openai_vision(b64_image, prompt)
            elif self.provider == "ollama":
                return self._call_ollama_vision(b64_image, prompt)
            else:
                logger.info(f"VLM credentials not set for {self.provider}; fallback to heuristic.")
                return YOLOv26Detector().classify_frame(jpeg_bytes)
        except Exception as exc:
            logger.error(f"VLM classification error: {exc}")
            return {
                "category": WasteCategory.NONE.value,
                "bin_index": -1,
                "confidence": 0.0,
            }

    def _call_gemini_vision(self, b64_image: str, prompt: str) -> dict[str, Any]:
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"response_mime_type": "application/json"},
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(raw_text)
                return self._format_result(data.get("category"), data.get("confidence", 0.90))
        return self._format_result("none", 0.0)

    def _call_openai_vision(self, b64_image: str, prompt: str) -> dict[str, Any]:
        import httpx

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                raw_text = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(raw_text)
                return self._format_result(data.get("category"), data.get("confidence", 0.90))
        return self._format_result("none", 0.0)

    def _call_ollama_vision(self, b64_image: str, prompt: str) -> dict[str, Any]:
        import httpx

        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": "llava",
            "prompt": prompt,
            "images": [b64_image],
            "format": "json",
            "stream": False,
        }
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                data = json.loads(resp.json().get("response", "{}"))
                return self._format_result(data.get("category"), data.get("confidence", 0.85))
        return self._format_result("none", 0.0)

    def _format_result(self, raw_category: str | None, confidence: float) -> dict[str, Any]:
        norm = (raw_category or "none").lower().strip()
        cat = LABEL_TO_CATEGORY_MAP.get(norm, WasteCategory.NONE)
        bin_idx = CATEGORY_TO_BIN_INDEX[cat]
        return {
            "category": cat.value,
            "bin_index": bin_idx,
            "confidence": round(float(confidence), 4),
        }


class MockDetector(BaseDetector):
    """Deterministic mock classifier for unit tests and verification."""

    def __init__(
        self,
        category: WasteCategory = WasteCategory.PLASTIC,
        confidence: float = 0.95,
    ) -> None:
        self.category = category
        self.confidence = confidence

    def set_mock(self, category: WasteCategory, confidence: float = 0.95) -> None:
        self.category = category
        self.confidence = confidence

    def classify_frame(self, jpeg_bytes: bytes) -> dict[str, Any]:
        is_valid, _, _ = validate_jpeg(jpeg_bytes)
        if not is_valid:
            return {
                "category": WasteCategory.NONE.value,
                "bin_index": -1,
                "confidence": 0.0,
            }
        return {
            "category": self.category.value,
            "bin_index": CATEGORY_TO_BIN_INDEX[self.category],
            "confidence": self.confidence,
        }


# Global detector singleton registry
_GLOBAL_DETECTOR: BaseDetector | None = None


def get_detector(backend_type: str | None = None) -> BaseDetector:
    """Factory function to get or instantiate detector backend."""
    global _GLOBAL_DETECTOR
    backend = (backend_type or os.getenv("MODEL_BACKEND", "yolov26")).lower()

    if _GLOBAL_DETECTOR is None:
        if backend in ("yolov26", "yolo", "yolov8"):
            _GLOBAL_DETECTOR = YOLOv26Detector()
        elif backend in ("vlm", "llm", "vision_llm"):
            _GLOBAL_DETECTOR = VisionLLMDetector()
        elif backend == "mock":
            _GLOBAL_DETECTOR = MockDetector()
        else:
            logger.warning(f"Unknown backend '{backend}', defaulting to YOLOv26Detector.")
            _GLOBAL_DETECTOR = YOLOv26Detector()

    return _GLOBAL_DETECTOR


def set_detector(detector: BaseDetector) -> None:
    """Set custom detector instance (useful for testing)."""
    global _GLOBAL_DETECTOR
    _GLOBAL_DETECTOR = detector


class PredictionSmoother:
    """Sliding-window majority-vote smoother to prevent frame-to-frame flicker.

    Maintains the last `window_size` predictions and outputs the category with the
    most votes (ties break toward the most recent frame so real object changes
    remain responsive while isolated single-frame glitches are filtered out).
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, int(window_size))
        self._history: deque[dict[str, Any]] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        """Clear prediction history."""
        self._history.clear()

    def update(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Record a raw prediction and return the temporally smoothed prediction."""
        if self.window_size <= 1:
            return dict(prediction)

        self._history.append(dict(prediction))

        counts: dict[str, int] = {}
        latest_idx: dict[str, int] = {}
        latest_pred: dict[str, dict[str, Any]] = {}
        for idx, item in enumerate(self._history):
            cat = str(item["category"])
            counts[cat] = counts.get(cat, 0) + 1
            latest_idx[cat] = idx
            latest_pred[cat] = item

        winner = max(counts.keys(), key=lambda c: (counts[c], latest_idx[c]))
        return dict(latest_pred[winner])


def classify_frame(jpeg_bytes: bytes) -> dict[str, Any]:
    """Top-level functional API for frame classification."""
    detector = get_detector()
    return detector.classify_frame(jpeg_bytes)
