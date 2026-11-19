"""Unit tests for SortSense detector module and model inference."""

import io

from PIL import Image

from app.detector import (
    LABEL_TO_CATEGORY_MAP,
    MockDetector,
    VisionLLMDetector,
    YOLOv26Detector,
    classify_frame,
    get_detector,
    set_detector,
    validate_jpeg,
)
from app.schemas import WasteCategory


def create_test_jpeg(width: int = 640, height: int = 480, color: tuple = (0, 0, 255)) -> bytes:
    """Helper to generate a valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_jpeg_valid():
    """Verify validation passes for valid 640x480 JPEG."""
    jpeg_bytes = create_test_jpeg(640, 480)
    is_valid, err_msg, dims = validate_jpeg(jpeg_bytes)
    assert is_valid is True
    assert err_msg is None
    assert dims == (640, 480)


def test_validate_jpeg_empty_payload():
    """Verify validation fails for empty or truncated payload."""
    is_valid, err_msg, dims = validate_jpeg(b"")
    assert is_valid is False
    assert "empty" in err_msg.lower()
    assert dims is None


def test_validate_jpeg_invalid_magic_bytes():
    """Verify validation fails when JPEG SOI marker (0xFF 0xD8) is absent."""
    non_jpeg = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    is_valid, err_msg, dims = validate_jpeg(non_jpeg)
    assert is_valid is False
    assert "marker" in err_msg.lower()
    assert dims is None


def test_validate_jpeg_corrupt_data():
    """Verify validation fails gracefully when payload starts with SOI but is corrupt."""
    corrupt = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00corrupted-tail-data"
    is_valid, err_msg, dims = validate_jpeg(corrupt)
    assert is_valid is False
    assert "failed to parse" in err_msg.lower()


def test_yolov26_detector_classify_valid_frame():
    """Verify YOLOv26Detector returns standardized contract on valid frame."""
    detector = YOLOv26Detector(confidence_threshold=0.5)
    # Blue image should trigger plastic heuristic in test mode
    jpeg_bytes = create_test_jpeg(640, 480, color=(0, 0, 240))
    result = detector.classify_frame(jpeg_bytes)

    assert "category" in result
    assert "bin_index" in result
    assert "confidence" in result
    assert result["category"] in ["plastic", "paper", "metal", "general", "none"]
    assert result["bin_index"] in [0, 1, 2, 3, -1]
    assert 0.0 <= result["confidence"] <= 1.0


def test_yolov26_detector_classify_dark_frame():
    """Verify dark/empty frame is classified as 'none' with bin_index -1."""
    detector = YOLOv26Detector()
    dark_jpeg = create_test_jpeg(640, 480, color=(5, 5, 5))
    result = detector.classify_frame(dark_jpeg)

    assert result["category"] == WasteCategory.NONE.value
    assert result["bin_index"] == -1
    assert result["confidence"] <= 0.20


def test_yolov26_detector_classify_invalid_bytes():
    """Verify invalid bytes return 'none' and -1 without raising exception."""
    detector = YOLOv26Detector()
    result = detector.classify_frame(b"random-bytes")
    assert result["category"] == WasteCategory.NONE.value
    assert result["bin_index"] == -1
    assert result["confidence"] == 0.0


def test_label_to_category_mapping_completeness():
    """Verify all standard recyclable classes exist in mapping."""
    assert LABEL_TO_CATEGORY_MAP["bottle"] == WasteCategory.PLASTIC
    assert LABEL_TO_CATEGORY_MAP["plastic_bag"] == WasteCategory.PLASTIC
    assert LABEL_TO_CATEGORY_MAP["paper"] == WasteCategory.PAPER
    assert LABEL_TO_CATEGORY_MAP["cardboard"] == WasteCategory.PAPER
    assert LABEL_TO_CATEGORY_MAP["can"] == WasteCategory.METAL
    assert LABEL_TO_CATEGORY_MAP["aluminum_can"] == WasteCategory.METAL
    assert LABEL_TO_CATEGORY_MAP["organic"] == WasteCategory.GENERAL
    assert LABEL_TO_CATEGORY_MAP["none"] == WasteCategory.NONE


def test_mock_detector():
    """Verify MockDetector returns configured category and supports dynamic reconfiguration."""
    detector = MockDetector(category=WasteCategory.METAL, confidence=0.99)
    jpeg_bytes = create_test_jpeg(640, 480)
    result = detector.classify_frame(jpeg_bytes)

    assert result["category"] == "metal"
    assert result["bin_index"] == 2
    assert result["confidence"] == 0.99

    detector.set_mock(WasteCategory.PAPER, 0.85)
    result2 = detector.classify_frame(jpeg_bytes)
    assert result2["category"] == "paper"
    assert result2["bin_index"] == 1
    assert result2["confidence"] == 0.85


def test_vision_llm_detector_fallback():
    """Verify VisionLLMDetector falls back cleanly when API keys are omitted."""
    vlm = VisionLLMDetector(provider="gemini")
    jpeg_bytes = create_test_jpeg(640, 480)
    result = vlm.classify_frame(jpeg_bytes)

    assert "category" in result
    assert result["bin_index"] in [0, 1, 2, 3, -1]


def test_global_detector_singleton():
    """Verify get_detector and set_detector singleton behaviour."""
    custom_mock = MockDetector(category=WasteCategory.GENERAL, confidence=0.77)
    set_detector(custom_mock)

    active = get_detector()
    assert active is custom_mock

    jpeg_bytes = create_test_jpeg(640, 480)
    result = classify_frame(jpeg_bytes)
    assert result["category"] == "general"
    assert result["bin_index"] == 3
    assert result["confidence"] == 0.77
