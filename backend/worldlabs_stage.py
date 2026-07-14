"""Optional World Labs image-to-world bridge. The API key never leaves the backend."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

API_ROOT = "https://api.worldlabs.ai/marble/v1"
ProgressCallback = Callable[[str], Awaitable[None]]

# Cost estimates are for one *non-pano image* input. They are displayed before
# a paid request and deliberately include the World Labs pano-generation event.
MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "marble-1.0-draft": {
        "label": "Marble 1.0 Draft",
        "estimated_credits": 230,
        "estimate_label": "about 230 credits per normal image",
        "quality": "fast credit check",
    },
    "marble-1.1": {
        "label": "Marble 1.1",
        "estimated_credits": 1580,
        "estimate_label": "about 1,580 credits per normal image",
        "quality": "recommended fixed-cost quality",
    },
    "marble-1.1-plus": {
        "label": "Marble 1.1 Plus",
        "estimated_credits": 1580,
        "max_estimated_credits": 3080,
        "estimate_label": "about 1,580 to 3,080 credits per normal image",
        "quality": "larger worlds, variable additional cost",
    },
}

_runtime_api_key: str | None = None
_runtime_model: str | None = None
_runtime_credit_balance: int | None = None


class WorldLabsError(RuntimeError):
    """A World Labs request failed or returned an incomplete world."""


def _env_api_key() -> str | None:
    return os.getenv("WORLDLABS_API_KEY") or os.getenv("WLT_API_KEY")


def _active_model() -> str:
    candidate = _runtime_model or os.getenv("WORLDLABS_MODEL", "marble-1.1")
    if candidate not in MODEL_OPTIONS:
        return "marble-1.1"
    return candidate


def is_configured() -> bool:
    return bool(_runtime_api_key or _env_api_key())


def settings_status() -> dict[str, Any]:
    """Return safe UI metadata only; the secret is never echoed to a client."""
    key_source = "session" if _runtime_api_key else ("environment" if _env_api_key() else None)
    model = _active_model()
    return {
        "available": is_configured(),
        "key_source": key_source,
        "session_only": key_source == "session",
        "model": model,
        "credit_balance": _runtime_credit_balance,
        "model_options": [{"id": model_id, **metadata} for model_id, metadata in MODEL_OPTIONS.items()],
        "input": "single non-panorama image",
    }


def clear_runtime_settings() -> None:
    """Forget a key entered through the local settings UI."""
    global _runtime_api_key, _runtime_model, _runtime_credit_balance
    _runtime_api_key = None
    _runtime_model = None
    _runtime_credit_balance = None


async def configure_runtime_settings(api_key: str, model: str) -> dict[str, Any]:
    """Verify a local-session key with the non-billable credits endpoint, then retain it in memory only."""
    global _runtime_api_key, _runtime_model, _runtime_credit_balance
    key = api_key.strip()
    if len(key) < 12:
        raise WorldLabsError("That World Labs API key is too short.")
    if model not in MODEL_OPTIONS:
        raise WorldLabsError("Unsupported World Labs model selection.")
    credit_balance = await _get_credit_balance(key)
    _runtime_api_key = key
    _runtime_model = model
    _runtime_credit_balance = credit_balance
    return settings_status()


async def _get_credit_balance(key: str) -> int | None:
    """Validate a key without creating a world or consuming generation credits."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        response = await client.get(f"{API_ROOT}/credits", headers={"WLT-Api-Key": key})
    if response.is_error:
        raise WorldLabsError(_error_message("Could not validate the World Labs API key", response))
    payload = response.json() or {}
    for field in ("remaining_credits", "credit_balance", "balance", "credits"):
        value = payload.get(field)
        if isinstance(value, (int, float)):
            return int(value)
    nested = payload.get("credits")
    if isinstance(nested, dict):
        for field in ("remaining", "balance", "available"):
            value = nested.get(field)
            if isinstance(value, (int, float)):
                return int(value)
    return None


def _api_key() -> str:
    key = _runtime_api_key or _env_api_key()
    if not key:
        raise WorldLabsError("World Mode needs a key in Settings or WORLDLABS_API_KEY in backend/.env.")
    return key


async def generate_world(
    image_path: Path,
    display_name: str,
    text_prompt: str,
    progress: ProgressCallback | None = None,
    model: str | None = None,
) -> dict[str, str]:
    """Upload one local image, start World Generation, and wait for its native world URL."""
    key = _api_key()
    headers = {"WLT-Api-Key": key, "Content-Type": "application/json"}
    extension = image_path.suffix.lstrip(".").lower() or "png"
    model = model or _active_model()
    if model not in MODEL_OPTIONS:
        raise WorldLabsError("Unsupported World Labs model selection.")
    timeout_sec = max(60, int(os.getenv("WORLDLABS_TIMEOUT_SECONDS", "900")))
    poll_seconds = max(3, int(os.getenv("WORLDLABS_POLL_SECONDS", "8")))

    async def say(message: str) -> None:
        logger.info("[WORLDLABS] %s", message)
        if progress:
            await progress(message)

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        await say("Preparing an encrypted media upload with World Labs...")
        prepare = await client.post(
            f"{API_ROOT}/media-assets:prepare_upload",
            headers=headers,
            json={"file_name": image_path.name, "kind": "image", "extension": extension},
        )
        if prepare.is_error:
            raise WorldLabsError(_error_message("Could not prepare the World Labs upload", prepare))
        prepared = prepare.json()
        upload = prepared.get("upload_info") or {}
        media = prepared.get("media_asset") or {}
        upload_url = upload.get("upload_url")
        media_id = media.get("id")
        if not upload_url or not media_id:
            raise WorldLabsError("World Labs returned incomplete media-upload details.")

        await say("Sending the selected frame to World Labs...")
        upload_headers = {str(k): str(v) for k, v in (upload.get("required_headers") or {}).items()}
        image_bytes = image_path.read_bytes()
        uploaded = await client.put(upload_url, headers=upload_headers, content=image_bytes)
        if uploaded.is_error:
            raise WorldLabsError(_error_message("Could not upload the selected frame", uploaded))

        world_prompt: dict[str, object] = {
            "type": "image",
            "image_prompt": {"source": "media_asset", "media_asset_id": media_id},
        }
        if text_prompt.strip():
            world_prompt["text_prompt"] = text_prompt.strip()

        await say("Opening a panoramic continuation from the source frame...")
        created = await client.post(
            f"{API_ROOT}/worlds:generate",
            headers=headers,
            json={"display_name": display_name[:120] or "LucidFrame World", "model": model, "world_prompt": world_prompt},
        )
        if created.is_error:
            raise WorldLabsError(_error_message("World Labs could not start this world", created))
        operation_id = (created.json() or {}).get("operation_id")
        if not operation_id:
            raise WorldLabsError("World Labs did not return an operation id.")

        deadline = time.monotonic() + timeout_sec
        last_description = ""
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_seconds)
            operation = await client.get(f"{API_ROOT}/operations/{operation_id}", headers={"WLT-Api-Key": key})
            if operation.is_error:
                raise WorldLabsError(_error_message("Could not check World Labs generation", operation))
            data = operation.json() or {}
            status = ((data.get("metadata") or {}).get("progress") or {})
            description = str(status.get("description") or status.get("status") or "World Labs is composing the 360-degree scene...")
            if description != last_description:
                await say(description)
                last_description = description
            if not data.get("done"):
                continue
            if data.get("error"):
                raise WorldLabsError(str(data["error"]))
            world = data.get("response") or {}
            world_url = world.get("world_marble_url")
            world_id = world.get("id") or ((data.get("metadata") or {}).get("world_id"))
            if not world_url:
                raise WorldLabsError("World Labs completed without a viewer URL.")
            assets = world.get("assets") or {}
            return {
                "world_id": str(world_id or ""),
                "world_url": str(world_url),
                "caption": str(assets.get("caption") or ""),
                "model": model,
            }

    raise WorldLabsError(f"World Labs did not finish within {timeout_sec // 60} minutes.")


def _error_message(prefix: str, response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text[:300]
    return f"{prefix} (HTTP {response.status_code}): {body}"
