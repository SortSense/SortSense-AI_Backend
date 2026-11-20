import asyncio
from pathlib import Path

import websockets
from PIL import Image

IMAGE_PATH = Path("test_image.jpg")


async def test():
    # If test_image.jpg doesn't exist, create a sample 640x480 JPEG frame automatically
    if not IMAGE_PATH.exists():
        print(f"'{IMAGE_PATH}' not found. Creating a sample 640x480 test JPEG...")
        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        img.save(IMAGE_PATH)

    with open(IMAGE_PATH, "rb") as f:
        jpeg_bytes = f.read()

    print(f"Connecting to ws://localhost:8000/ws/vision (sending {len(jpeg_bytes)} bytes)...")
    async with websockets.connect("ws://localhost:8000/ws/vision?station_id=ST-001") as ws:
        await ws.send(jpeg_bytes)
        response = await ws.recv()
        print("\n--> Live Model Prediction from Backend:")
        print(response)


if __name__ == "__main__":
    asyncio.run(test())
