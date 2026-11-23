"""SortSense Webcam Vision & AI Inference Testing Tool.

Captures live video from the laptop webcam, matches the exact resolution and
JPEG compression quality configured on the ESP32-S3 hardware (OV5640 camera),
and runs inference through the AI classification backend.

Supports:
1. Direct in-process classification via `app.detector.classify_frame`
2. Live streaming over WebSocket to `/ws/vision` (simulating hardware client)
3. On-screen HUD displaying predicted bin category, bin index, LED color,
   confidence score, payload size, latency, and FPS.
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

# Automatically switch to project virtual environment if available (Windows & POSIX)
_VENV_DIR = Path(__file__).resolve().parent.parent / ".venv"
_VENV_PYTHON = _VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if _VENV_PYTHON.exists() and Path(sys.executable).resolve() != _VENV_PYTHON.resolve():
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.detector import PredictionSmoother, classify_frame, get_detector  # noqa: E402
from app.schemas import WasteCategory  # noqa: E402

# Color mappings (BGR format for OpenCV drawing)
BIN_COLORS_BGR = {
    WasteCategory.PLASTIC: (255, 128, 0),  # Blue
    WasteCategory.PAPER: (0, 200, 0),  # Green
    WasteCategory.METAL: (0, 215, 255),  # Yellow/Gold
    WasteCategory.GENERAL: (255, 255, 255),  # White
    WasteCategory.NONE: (200, 200, 200),  # Soft Gray/Cyan
}

# Terminal ANSI color codes for headless / console mode
ANSI_COLORS = {
    WasteCategory.PLASTIC: "\033[94m",  # Blue
    WasteCategory.PAPER: "\033[92m",  # Green
    WasteCategory.METAL: "\033[93m",  # Yellow
    WasteCategory.GENERAL: "\033[97m",  # White
    WasteCategory.NONE: "\033[90m",  # Gray
}
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"


def encode_frame_esp32_format(
    bgr_frame: np.ndarray,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
) -> tuple[bytes, np.ndarray]:
    """Resize and compress a frame to match the ESP32-S3 OV5640 JPEG pipeline.

    Note on JPEG quality scales:
    ESP32 `frame2jpg` (`CAM_JPEG_QUALITY=12`) uses an inverted 0-63 scale where
    lower = sharper (~80-85% standard quality). Python Pillow uses the standard
    1-95 scale where higher = sharper, so Pillow `quality=80` matches ESP32 `12`.
    """
    # Resize to target camera resolution
    resized = cv2.resize(bgr_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Convert BGR (OpenCV) to RGB (PIL) for accurate JPEG compression
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Encode to JPEG bytes
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality)
    jpeg_bytes = buf.getvalue()

    # Decode compressed JPEG for preview so the HUD shows exactly what YOLO sees
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    preview_bgr = decoded if decoded is not None else resized

    return jpeg_bytes, preview_bgr


class BoxSmoother:
    """EMA smoother for the primary handheld bounding box to eliminate visual jitter."""

    def __init__(self, alpha: float = 0.35, hold_frames: int = 3) -> None:
        self.alpha = alpha
        self.hold_frames = hold_frames
        self._xyxy: list[float] | None = None
        self._missed = 0

    def update(self, primary_det: dict | None, active_category: str) -> list[float] | None:
        """Update smoothed coordinates; holds briefly across 1-2 dropped frames."""
        if active_category == WasteCategory.NONE.value:
            self._xyxy = None
            self._missed = 0
            return None

        xyxy = primary_det.get("xyxy") if primary_det else None
        if xyxy and len(xyxy) == 4:
            self._missed = 0
            if self._xyxy is None:
                self._xyxy = [float(v) for v in xyxy]
            else:
                a = self.alpha
                self._xyxy = [a * float(n) + (1.0 - a) * p for n, p in zip(xyxy, self._xyxy)]
            return self._xyxy

        self._missed += 1
        if self._xyxy is not None and self._missed <= self.hold_frames:
            return self._xyxy

        self._xyxy = None
        return None


def draw_scan_zone_guide(
    frame: np.ndarray,
    roi_bounds: tuple[float, float, float, float],
    active: bool,
) -> np.ndarray:
    """Draw subtle corner brackets showing the center handheld scan zone."""
    h, w, _ = frame.shape
    rx1, ry1, rx2, ry2 = roi_bounds
    x1, y1, x2, y2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
    corner_len = 22
    color = (110, 230, 140) if active else (150, 150, 150)
    thickness = 2

    # 4 clean viewfinder corner brackets
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    if not active:
        cv2.putText(
            frame,
            "HOLD ITEM INSIDE FRAME",
            (x1 + 10, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
    return frame


def draw_primary_box(
    frame: np.ndarray,
    smooth_xyxy: list[float] | None,
    category_name: str,
    confidence: float,
) -> np.ndarray:
    """Draw only the single smoothed foreground object bounding box."""
    if not smooth_xyxy or category_name == WasteCategory.NONE.value:
        return frame

    x1, y1, x2, y2 = (int(round(v)) for v in smooth_xyxy)
    try:
        cat_enum = WasteCategory(category_name)
    except ValueError:
        cat_enum = WasteCategory.NONE

    color = BIN_COLORS_BGR.get(cat_enum, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{category_name.upper()} {int(confidence * 100)}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_y = max(92 + th, y1 - 6)
    cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, label_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return frame


def draw_hud_overlay(
    display_frame: np.ndarray,
    category_name: str,
    bin_index: int,
    confidence: float,
    latency_ms: float,
    fps: float,
    jpeg_bytes_len: int,
    resolution: tuple[int, int],
    jpeg_quality: int,
    mode: str,
) -> np.ndarray:
    """Render a clean status Heads-Up Display (HUD) overlay on the preview frame."""
    frame = display_frame.copy()
    h, w, _ = frame.shape

    # Determine category enum
    try:
        cat_enum = WasteCategory(category_name)
    except ValueError:
        cat_enum = WasteCategory.NONE

    badge_color = BIN_COLORS_BGR.get(cat_enum, (200, 200, 200))

    # Top banner background (semi-transparent dark overlay)
    overlay = frame.copy()
    banner_height = 85
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status LED color pill on the top left
    led_radius = 16
    cv2.circle(frame, (28, 42), led_radius, badge_color, -1)
    cv2.circle(frame, (28, 42), led_radius, (255, 255, 255), 2)

    # Category title and bin index
    cat_text = category_name.upper()
    if bin_index >= 0:
        bin_text = f"BIN {bin_index}"
    else:
        bin_text = "STANDBY"

    cv2.putText(
        frame,
        f"{cat_text} ({bin_text})",
        (55, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        badge_color,
        2,
        cv2.LINE_AA,
    )

    # Confidence bar and text
    conf_percent = int(confidence * 100)
    conf_str = f"Confidence: {conf_percent}%"
    cv2.putText(
        frame,
        conf_str,
        (55, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Confidence progress bar
    bar_x = 220
    bar_y = 52
    bar_w = 120
    bar_h = 12
    fill_w = int(bar_w * min(1.0, max(0.0, confidence)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), badge_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    # Right side: hardware emulation metrics
    kb_size = jpeg_bytes_len / 1024.0
    line1 = f"Res: {resolution[0]}x{resolution[1]} | Q: {jpeg_quality} | {kb_size:.1f} KB"
    line2 = f"Latency: {latency_ms:.1f} ms | FPS: {fps:.1f} | Mode: {mode.upper()}"

    cv2.putText(
        frame,
        line1,
        (w - 340, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        line2,
        (w - 340, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (160, 220, 160),
        1,
        cv2.LINE_AA,
    )

    # Bottom helper bar
    bottom_overlay = frame.copy()
    cv2.rectangle(bottom_overlay, (0, h - 28), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(bottom_overlay, 0.7, frame, 0.3, 0, frame)
    helper_text = "Press 'q' to quit | 's' to save snapshot | 'space' to pause"
    cv2.putText(
        frame,
        helper_text,
        (12, h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return frame


def print_console_metric(
    category_name: str,
    bin_index: int,
    confidence: float,
    latency_ms: float,
    fps: float,
    bytes_len: int,
) -> None:
    """Print clean ANSI status line for headless or terminal monitoring."""
    try:
        cat_enum = WasteCategory(category_name)
    except ValueError:
        cat_enum = WasteCategory.NONE

    color = ANSI_COLORS.get(cat_enum, ANSI_RESET)
    conf_pct = confidence * 100
    kb = bytes_len / 1024.0

    msg = (
        f"[{color}{ANSI_BOLD}{category_name.upper():<8}{ANSI_RESET}] "
        f"Bin: {bin_index:>2} | "
        f"Conf: {conf_pct:>5.1f}% | "
        f"Latency: {latency_ms:>5.1f}ms | "
        f"FPS: {fps:>4.1f} | "
        f"Payload: {kb:>4.1f} KB"
    )
    sys.stdout.write(f"\r{msg}")
    sys.stdout.flush()


def run_direct_mode(
    camera_index: int,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
    no_gui: bool,
    max_frames: int | None = None,
    smooth_window: int = 5,
    full_frame: bool = False,
) -> None:
    """Run camera inference directly against in-process classifier."""
    print("=" * 70)
    print("SortSense AI Webcam Test (Direct In-Process Mode)")
    print(f"Target Camera Resolution : {target_width}x{target_height}")
    print(f"JPEG Quality (Pillow)    : {jpeg_quality} (80 ~= ESP32 CAM_JPEG_QUALITY 12)")
    print(f"Temporal Smoothing       : {smooth_window}-frame majority vote")
    print(f"Handheld Center Focus    : {'OFF (full frame)' if full_frame else 'ON (center zone)'}")
    print(f"Camera Device Index      : {camera_index}")
    print("=" * 70)

    detector = get_detector()
    if hasattr(detector, "focus_center_roi"):
        detector.focus_center_roi = not full_frame
    backend_name = type(detector).__name__
    model_loaded = getattr(detector, "model", None) is not None
    weights_path = getattr(detector, "weights_path", "N/A")
    print(f"Active Vision Backend: {backend_name}")
    if model_loaded:
        print(f"Model Weights Loaded : {weights_path}")
    else:
        print("WARNING: No YOLO weights loaded — running fallback color heuristic!")
    print("Starting webcam capture... Press 'q' or Ctrl+C to exit.\n")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera on index {camera_index}.")
        return

    # Request desired capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    paused = False
    last_time = time.monotonic()
    frame_count = 0
    fps = 0.0
    smoother = PredictionSmoother(window_size=smooth_window)
    box_smoother = BoxSmoother(alpha=0.35, hold_frames=3)
    smooth_box: list[float] | None = None

    last_prediction = {
        "category": WasteCategory.NONE.value,
        "bin_index": -1,
        "confidence": 0.0,
    }
    last_latency = 0.0
    last_jpeg_len = 0

    try:
        while True:
            if max_frames and frame_count >= max_frames:
                break

            if not paused:
                ret, raw_frame = cap.read()
                if not ret:
                    print("\nWarning: Dropped webcam frame.")
                    time.sleep(0.05)
                    continue

                # Encode to match exact ESP32-S3 format
                jpeg_bytes, resized_frame = encode_frame_esp32_format(
                    raw_frame, target_width, target_height, jpeg_quality
                )
                last_jpeg_len = len(jpeg_bytes)

                # Run inference + temporal smoothing
                t0 = time.monotonic()
                raw_prediction = classify_frame(jpeg_bytes)
                last_prediction = smoother.update(raw_prediction)
                smooth_box = box_smoother.update(
                    getattr(detector, "last_primary_detection", None),
                    last_prediction["category"],
                )
                t1 = time.monotonic()
                last_latency = (t1 - t0) * 1000.0

                frame_count += 1
                now = time.monotonic()
                dt = now - last_time
                if dt >= 0.5:
                    fps = 1.0 / max(0.001, (dt / max(1, frame_count)))
                    frame_count = 0
                    last_time = now

            if no_gui:
                print_console_metric(
                    last_prediction["category"],
                    int(last_prediction["bin_index"]),
                    float(last_prediction["confidence"]),
                    last_latency,
                    fps,
                    last_jpeg_len,
                )
                time.sleep(0.03)
            else:
                preview = resized_frame.copy()
                roi_bounds = getattr(detector, "roi_bounds", (0.18, 0.18, 0.82, 0.88))
                if not full_frame:
                    draw_scan_zone_guide(
                        preview,
                        roi_bounds,
                        active=last_prediction["category"] != WasteCategory.NONE.value,
                    )
                draw_primary_box(
                    preview,
                    smooth_box,
                    last_prediction["category"],
                    float(last_prediction["confidence"]),
                )
                annotated = draw_hud_overlay(
                    preview,
                    last_prediction["category"],
                    int(last_prediction["bin_index"]),
                    float(last_prediction["confidence"]),
                    last_latency,
                    fps,
                    last_jpeg_len,
                    (target_width, target_height),
                    jpeg_quality,
                    mode="Direct",
                )
                cv2.imshow("SortSense AI Vision Test", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break
                elif key == ord(" "):  # Spacebar to pause/resume
                    paused = not paused
                    print(f"\nStream {'PAUSED' if paused else 'RESUMED'}")
                elif key == ord("s"):  # Snapshot
                    snap_filename = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_filename, resized_frame)
                    json_filename = f"snapshot_{int(time.time())}.json"
                    with open(json_filename, "w") as jf:
                        json.dump(last_prediction, jf, indent=2)
                    print(f"\nSaved snapshot to '{snap_filename}' and '{json_filename}'")

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()
        print("\nWebcam released cleanly.")


async def run_websocket_mode(
    camera_index: int,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
    ws_url: str,
    no_gui: bool,
    max_frames: int | None = None,
) -> None:
    """Stream camera frames over WebSocket, exactly emulating ESP32-S3."""
    import websockets

    print("=" * 70)
    print("SortSense AI Webcam Test (WebSocket Client Simulation Mode)")
    print(f"Target Camera Resolution : {target_width}x{target_height}")
    print(f"JPEG Quality Compression : {jpeg_quality} (matches ESP32 CAM_JPEG_QUALITY)")
    print(f"WebSocket Target URL     : {ws_url}")
    print("=" * 70)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera on index {camera_index}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

    print(f"Connecting to {ws_url}...")
    try:
        async with websockets.connect(ws_url) as ws:
            print("Connected to SortSense Vision WebSocket successfully!")
            print("Streaming live frames... Press 'q' or Ctrl+C to stop.\n")

            frame_count = 0
            last_time = time.monotonic()
            fps = 0.0

            while True:
                if max_frames and frame_count >= max_frames:
                    break

                ret, raw_frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.05)
                    continue

                jpeg_bytes, resized_frame = encode_frame_esp32_format(
                    raw_frame, target_width, target_height, jpeg_quality
                )

                t0 = time.monotonic()
                # Send raw binary JPEG bytes
                await ws.send(jpeg_bytes)
                # Receive JSON classification response
                raw_response = await ws.recv()
                t1 = time.monotonic()
                latency_ms = (t1 - t0) * 1000.0

                try:
                    prediction = json.loads(raw_response)
                except Exception:
                    prediction = {"category": "none", "bin_index": -1, "confidence": 0.0}

                frame_count += 1
                now = time.monotonic()
                dt = now - last_time
                if dt >= 0.5:
                    fps = 1.0 / max(0.001, (dt / max(1, frame_count)))
                    frame_count = 0
                    last_time = now

                if no_gui:
                    print_console_metric(
                        prediction.get("category", "none"),
                        int(prediction.get("bin_index", -1)),
                        float(prediction.get("confidence", 0.0)),
                        latency_ms,
                        fps,
                        len(jpeg_bytes),
                    )
                    await asyncio.sleep(0.02)
                else:
                    annotated = draw_hud_overlay(
                        resized_frame,
                        prediction.get("category", "none"),
                        int(prediction.get("bin_index", -1)),
                        float(prediction.get("confidence", 0.0)),
                        latency_ms,
                        fps,
                        len(jpeg_bytes),
                        (target_width, target_height),
                        jpeg_quality,
                        mode="WebSocket",
                    )
                    cv2.imshow("SortSense AI Vision Test (WebSocket)", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

    except Exception as exc:
        print(f"\nWebSocket connection error: {exc}")
        print("Tip: Make sure the backend server is running: `uvicorn app.main:app --port 8000`")
    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()
        print("\nWebcam and WebSocket released cleanly.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SortSense Laptop Webcam AI Testing & Resolution Matching Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "ws"],
        default="direct",
        help="Inference mode: 'direct' runs model in-process; 'ws' streams over WebSocket.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Webcam device index (/dev/video0 is 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Target frame width (640 for VGA or 480 for HVGA).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Target frame height (480 for VGA or 320 for HVGA).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="Pillow JPEG quality (1-95, higher=sharper; 80 matches ESP32 CAM_JPEG_QUALITY=12).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Temporal majority-vote window size in direct mode (1 disables smoothing).",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Disable center handheld scan zone and detect across the entire frame.",
    )
    parser.add_argument(
        "--ws-url",
        type=str,
        default="ws://localhost:8000/ws/vision?station_id=ST-001",
        help="Target WebSocket endpoint for 'ws' mode.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable OpenCV preview window and use terminal status output.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Optional limit on number of frames to capture (useful for automated tests).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_arguments()

    if args.mode == "direct":
        run_direct_mode(
            camera_index=args.device,
            target_width=args.width,
            target_height=args.height,
            jpeg_quality=args.quality,
            no_gui=args.no_gui,
            max_frames=args.frames,
            smooth_window=args.smooth,
            full_frame=args.full_frame,
        )
    elif args.mode == "ws":
        asyncio.run(
            run_websocket_mode(
                camera_index=args.device,
                target_width=args.width,
                target_height=args.height,
                jpeg_quality=args.quality,
                ws_url=args.ws_url,
                no_gui=args.no_gui,
                max_frames=args.frames,
            )
        )


if __name__ == "__main__":
    main()
