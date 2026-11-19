"""Async HTTP Client for Dispatching SortSense Telemetry.

Posts verified waste sorting events to the SortSense Cloud Website API:
  POST {CLOUD_API_URL}/api/events/log-waste
"""

import asyncio
import logging
import os

import httpx

from app.schemas import CloudTelemetryPayload

logger = logging.getLogger(__name__)


class CloudClient:
    """Asynchronous client responsible for forwarding verified sort telemetry to the Cloud."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = (base_url or os.getenv("CLOUD_API_URL", "http://localhost:3000")).rstrip(
            "/"
        )
        self.timeout = float(os.getenv("CLOUD_API_TIMEOUT", str(timeout)))
        self.max_retries = int(os.getenv("CLOUD_API_MAX_RETRIES", str(max_retries)))
        self._client: httpx.AsyncClient | None = None

    async def get_client(self) -> httpx.AsyncClient:
        """Return or lazily create httpx AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "SortSense-AI_Backend/1.0.0",
                },
            )
        return self._client

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def post_waste_event(self, payload: CloudTelemetryPayload) -> bool:
        """Dispatch waste classification and drop telemetry to SortSense Cloud API.

        Retries with exponential backoff on transient network failures.
        Returns:
            True if event was successfully accepted (HTTP 2xx), False otherwise.
        """
        endpoint = f"{self.base_url}/api/events/log-waste"
        data = payload.model_dump()
        client = await self.get_client()

        backoff = 0.3
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Dispatching telemetry (attempt {attempt}/{self.max_retries}) to {endpoint}: {data}"
                )
                response = await client.post("/api/events/log-waste", json=data)

                if 200 <= response.status_code < 300:
                    logger.info(
                        f"Telemetry delivered successfully to {endpoint} (HTTP {response.status_code})"
                    )
                    return True
                else:
                    logger.warning(
                        f"Cloud API returned HTTP {response.status_code} on attempt {attempt}: {response.text}"
                    )
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                logger.warning(f"Connection error reaching Cloud API on attempt {attempt}: {exc}")

            if attempt < self.max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2

        logger.error(
            f"Failed to deliver telemetry to Cloud API after {self.max_retries} attempts: {data}"
        )
        return False


# Global cloud client instance
_cloud_client: CloudClient | None = None


def get_cloud_client() -> CloudClient:
    """Return global CloudClient instance."""
    global _cloud_client
    if _cloud_client is None:
        _cloud_client = CloudClient()
    return _cloud_client


def set_cloud_client(client: CloudClient) -> None:
    """Override global CloudClient instance (useful for testing)."""
    global _cloud_client
    _cloud_client = client
