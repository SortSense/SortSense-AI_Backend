"""Unit tests for CloudClient async HTTP client."""

import httpx
import pytest

from app.cloud_client import CloudClient, get_cloud_client, set_cloud_client
from app.schemas import CloudTelemetryPayload


@pytest.mark.asyncio
async def test_cloud_client_successful_dispatch():
    """Verify successful HTTP 200 response from Cloud API returns True."""
    payload = CloudTelemetryPayload(
        station_id="ST-001",
        waste_category="plastic",
        dropped_in_bin="plastic",
        confidence=0.91,
        is_correct_sort=True,
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/api/events/log-waste")
        return httpx.Response(200, json={"status": "event_logged", "id": 123})

    transport = httpx.MockTransport(mock_handler)
    client = CloudClient(base_url="http://mock-cloud.local", max_retries=1)
    # Inject mock client with transport
    client._client = httpx.AsyncClient(base_url="http://mock-cloud.local", transport=transport)

    success = await client.post_waste_event(payload)
    assert success is True
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_server_error_and_retry():
    """Verify CloudClient retries on 500 and returns False if all attempts fail."""
    payload = CloudTelemetryPayload(
        station_id="ST-002",
        waste_category="metal",
        dropped_in_bin="metal",
        confidence=0.88,
        is_correct_sort=True,
    )

    call_count = 0

    def error_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(500, json={"error": "Database unavailable"})

    transport = httpx.MockTransport(error_handler)
    client = CloudClient(base_url="http://mock-cloud.local", max_retries=2)
    client._client = httpx.AsyncClient(base_url="http://mock-cloud.local", transport=transport)

    success = await client.post_waste_event(payload)
    assert success is False
    assert call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_cloud_client_network_exception():
    """Verify CloudClient catches network exceptions and does not crash."""
    payload = CloudTelemetryPayload(
        station_id="ST-003",
        waste_category="paper",
        dropped_in_bin="plastic",
        confidence=0.75,
        is_correct_sort=False,
    )

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connection timed out", request=request)

    transport = httpx.MockTransport(timeout_handler)
    client = CloudClient(base_url="http://mock-cloud.local", max_retries=2)
    client._client = httpx.AsyncClient(base_url="http://mock-cloud.local", transport=transport)

    success = await client.post_waste_event(payload)
    assert success is False
    await client.close()


def test_cloud_client_singleton_getter_setter():
    """Verify get_cloud_client and set_cloud_client."""
    custom = CloudClient(base_url="http://custom.local")
    set_cloud_client(custom)
    active = get_cloud_client()
    assert active is custom
    assert active.base_url == "http://custom.local"
