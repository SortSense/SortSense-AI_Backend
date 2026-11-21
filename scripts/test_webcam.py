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

import os
import sys
from pathlib import Path

# Automatically switch to project virtual environment if available
_VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "python"
if _VENV_PYTHON.exists() and sys.executable != str(_VENV_PYTHON):
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

import argparse
import asyncio
import io
import json
import time

import cv2
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.detector import classify_frame, get_detector
from app.schemas import CATEGORY_TO_BIN_INDEX, WasteCategory

# Color mappings (BGR format for OpenCV drawing)
BIN_COLORS_BGR = {
    WasteCategory.PLASTIC: (255, 128, 0),     # Blue
    WasteCategory.PAPER: (0, 200, 0),         # Green
    WasteCategory.METAL: (0, 215, 255),       # Yellow/Gold
    WasteCategory.GENERAL: (255, 255, 255),   # White
    WasteCategory.NONE: (200, 200, 200),      # Soft Gray/Cyan
}

# Terminal ANSI color codes for headless / console mode
ANSI_COLORS = {
    WasteCategory.PLASTIC: "\033[94m",   # Blue
    WasteCategory.PAPER: "\033[92m",     # Green
    WasteCategory.METAL: "\033[93m",     # Yellow
    WasteCategory.GENERAL: "\033[97m",   # White
    WasteCategory.NONE: "\033[90m",      # Gray
}
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"


def encode_frame_esp32_format(
    bgr_frame: np.ndarray,
    target_width: int,
    target_height: int,
    jpeg_quality: int,
) -> tuple[bytes, np.ndarray]:
    """Resize and compress a frame to match the exact ESP32-S3 OV5640 pipeline.

    The ESP32-S3 captures in native YUV422 format and uses `frame2jpg` in PSRAM
    with `CAM_JPEG_QUALITY` (default 12) before network transmission.
    """
    # Resize to target camera resolution
    resized = cv2.resize(bgr_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Convert BGR (OpenCV) to RGB (PIL) for accurate JPEG compression
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Encode to JPEG bytes using the identical quality setting
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=jpeg_quality)
    jpeg_bytes = buf.getvalue()

    return jpeg_bytes, resized


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
) -> None:
    """Run camera inference directly against in-process classifier."""
    print("=" * 70)
    print("SortSense AI Webcam Test (Direct In-Process Mode)")
    print(f"Target Camera Resolution : {target_width}x{target_height}")
    print(f"JPEG Quality Compression : {jpeg_quality} (matches ESP32 CAM_JPEG_QUALITY)")
    print(f"Camera Device Index      : {camera_index}")
    print("=" * 70)

    detector = get_detector()
    backend_name = type(detector).__name__
    print(f"Active Vision Backend: {backend_name}")
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

                # Run inference
                t0 = time.monotonic()
                last_prediction = classify_frame(jpeg_bytes)
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
                annotated = draw_hud_overlay(
                    resized_frame,
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
        default=12,
        help="JPEG quality factor (matches ESP32 CAM_JPEG_QUALITY in config.h).",
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
