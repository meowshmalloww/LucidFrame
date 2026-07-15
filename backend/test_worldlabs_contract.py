"""Cost-free contract tests for the World Labs bridge.

All HTTP exchanges are mocked.  These tests must never start a generation or
consume API credits.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

import worldlabs_stage

_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _Client:
    """AsyncClient-compatible wrapper around an in-memory MockTransport."""

    def __init__(self, *args, **kwargs) -> None:
        self._client = _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(self._handle))
        self.requests: list[httpx.Request] = []

    async def __aenter__(self):
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args):
        return await self._client.__aexit__(*args)

    async def get(self, *args, **kwargs):
        return await self._client.get(*args, **kwargs)

    async def post(self, *args, **kwargs):
        return await self._client.post(*args, **kwargs)

    async def put(self, *args, **kwargs):
        return await self._client.put(*args, **kwargs)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/credits"):
            return httpx.Response(200, json={"credits": 6250})
        if request.url.path.endswith("/media-assets:prepare_upload"):
            return httpx.Response(
                200,
                json={
                    "media_asset": {
                        "media_asset_id": "media-123",
                        "file_name": "frame.png",
                        "kind": "image",
                        "created_at": "2026-01-01T00:00:00Z",
                    },
                    "upload_info": {
                        "upload_method": "PUT",
                        "upload_url": "https://upload.invalid/frame",
                        "required_headers": {"Content-Type": "image/png"},
                    },
                },
            )
        if request.url.host == "upload.invalid":
            return httpx.Response(200)
        if request.url.path.endswith("/worlds:generate"):
            payload = __import__("json").loads(request.content)
            image_prompt = payload["world_prompt"]["image_prompt"]
            assert image_prompt["media_asset_id"] == "media-123"
            assert payload["world_prompt"]["is_pano"] is False
            assert len(payload["display_name"]) <= 64
            return httpx.Response(200, json={"operation_id": "operation-123"})
        if request.url.path.endswith("/operations/operation-123"):
            return httpx.Response(
                200,
                json={
                    "done": True,
                    "metadata": {"progress": 100},
                    "response": {
                        "world_id": "world-123",
                        "world_marble_url": "https://marble.worldlabs.ai/world/world-123",
                        "assets": {"caption": "A spatial frame"},
                    },
                },
            )
        return httpx.Response(404, json={"detail": "unexpected mocked URL"})


@pytest.fixture(autouse=True)
def _clear_settings():
    worldlabs_stage.clear_runtime_settings()
    yield
    worldlabs_stage.clear_runtime_settings()


def test_settings_validation_uses_documented_credit_response(monkeypatch) -> None:
    monkeypatch.setattr(worldlabs_stage.httpx, "AsyncClient", _Client)
    status = asyncio.run(
        worldlabs_stage.configure_runtime_settings("wlt_test_key_123456", "marble-1.1")
    )
    assert status["available"] is True
    assert status["credit_balance"] == 6250


def test_generate_world_uses_public_media_and_world_ids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(worldlabs_stage.httpx, "AsyncClient", _Client)
    worldlabs_stage._runtime_api_key = "wlt_test_key_123456"
    image = tmp_path / "frame.png"
    image.write_bytes(b"not-decoded-by-the-api-bridge")

    async def run():
        return await worldlabs_stage.generate_world(
            image,
            "x" * 100,
            "continue the photographed room",
            model="marble-1.1",
        )

    # Avoid patching asyncio.sleep recursively; the first operation response is
    # already complete, so make the configured poll effectively immediate.
    monkeypatch.setenv("WORLDLABS_POLL_SECONDS", "3")
    original_sleep = asyncio.sleep

    async def immediate_sleep(_seconds):
        await original_sleep(0)

    monkeypatch.setattr(worldlabs_stage.asyncio, "sleep", immediate_sleep)
    result = asyncio.run(run())
    assert result["world_id"] == "world-123"
    assert result["world_url"].endswith("/world-123")
    assert result["caption"] == "A spatial frame"
