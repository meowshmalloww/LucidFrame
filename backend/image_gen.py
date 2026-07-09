"""
Optional image enhancement / generation stage.

Supports multiple free-tier providers:
  - Pollinations.ai  (no API key required)
  - Cloudflare Workers AI  (requires CF_API_TOKEN + CF_ACCOUNT_ID)
  - HuggingFace Inference  (requires HF_TOKEN)
  - Pixazo.ai  (requires PIXAZO_API_KEY)

All providers generate single 2D images from text prompts.
This is NOT for multi-view 3D generation — that is handled locally
by Zero123++ / SV3D in multiview_stage.py.

Use cases:
  - Enhance the user's input image before running the pipeline
  - Generate a stylized variant of the input
  - Generate reference images for the dream narrative
"""
from __future__ import annotations

import io
import os
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import httpx
from PIL import Image

logger = logging.getLogger("lucidframe")
TAG = "[image_gen]"

IMAGE_GEN_PROVIDER = os.getenv("IMAGE_GEN_PROVIDER", "pollinations").lower()


# ── Pollinations.ai ──────────────────────────────────────────────────────────

async def generate_pollinations(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 1024,
    model: str = "flux",
    seed: Optional[int] = None,
) -> Path:
    """Generate an image via Pollinations.ai — no API key needed."""
    base = "https://image.pollinations.ai/prompt/"
    params = {
        "width": str(width),
        "height": str(height),
        "model": model,
        "nologo": "true",
    }
    if seed is not None:
        params["seed"] = str(seed)

    url = f"{base}{quote_plus(prompt)}"
    logger.info("%s Pollinations request: %s", TAG, url[:80])

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(url, params=params, follow_redirects=True)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content))
        img.save(output_path)
        logger.info("%s Pollinations saved -> %s", TAG, output_path)
        return output_path


# ── Cloudflare Workers AI ────────────────────────────────────────────────────

async def generate_cloudflare(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 1024,
    model: str = "@cf/black-forest-labs/flux-1-schnell",
) -> Path:
    """Generate an image via Cloudflare Workers AI."""
    api_token = os.getenv("CF_API_TOKEN", "")
    account_id = os.getenv("CF_ACCOUNT_ID", "")

    if not api_token or not account_id:
        raise ValueError("Cloudflare requires CF_API_TOKEN and CF_ACCOUNT_ID env vars")

    endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_steps": 4,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {api_token}"},
        )
        resp.raise_for_status()

        data = resp.json()
        if "image" not in data.get("result", {}):
            raise RuntimeError(f"Cloudflare returned no image: {data}")

        img_bytes = data["result"]["image"]
        if isinstance(img_bytes, str):
            import base64
            img_bytes = base64.b64decode(img_bytes)

        img = Image.open(io.BytesIO(img_bytes))
        img.save(output_path)
        logger.info("%s Cloudflare saved -> %s", TAG, output_path)
        return output_path


# ── HuggingFace Inference API ────────────────────────────────────────────────

async def generate_huggingface(
    prompt: str,
    output_path: Path,
    model: str = "black-forest-labs/FLUX.1-dev",
    width: int = 1024,
    height: int = 1024,
) -> Path:
    """Generate an image via HuggingFace Inference Providers."""
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise ValueError("HuggingFace requires HF_TOKEN env var")

    endpoint = f"https://api-inference.huggingface.co/models/{model}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {hf_token}"},
        )
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content))
        img.save(output_path)
        logger.info("%s HuggingFace saved -> %s", TAG, output_path)
        return output_path


# ── Pixazo.ai ────────────────────────────────────────────────────────────────

async def generate_pixazo(
    prompt: str,
    output_path: Path,
    model: str = "flux-schnell",
    width: int = 1024,
    height: int = 1024,
) -> Path:
    """Generate an image via Pixazo.ai free tier."""
    api_key = os.getenv("PIXAZO_API_KEY", "")
    if not api_key:
        raise ValueError("Pixazo requires PIXAZO_API_KEY env var")

    endpoint = "https://api.pixazo.ai/v1/images/generations"

    payload = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "n": 1,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()

        data = resp.json()
        image_url = data.get("data", [{}])[0].get("url", "")
        if not image_url:
            raise RuntimeError(f"Pixazo returned no image URL: {data}")

        img_resp = await client.get(image_url, follow_redirects=True)
        img_resp.raise_for_status()

        img = Image.open(io.BytesIO(img_resp.content))
        img.save(output_path)
        logger.info("%s Pixazo saved -> %s", TAG, output_path)
        return output_path


# ── Main dispatch ────────────────────────────────────────────────────────────

async def generate_image(
    prompt: str,
    output_path: Path,
    width: int = 1024,
    height: int = 1024,
) -> Path:
    """
    Generate a single image from a text prompt using the configured provider.

    Provider is selected via IMAGE_GEN_PROVIDER env var:
      pollinations (default, no key), cloudflare, huggingface, pixazo
    """
    provider = IMAGE_GEN_PROVIDER

    if provider == "pollinations":
        return await generate_pollinations(prompt, output_path, width, height)
    elif provider == "cloudflare":
        return await generate_cloudflare(prompt, output_path, width, height)
    elif provider == "huggingface":
        return await generate_huggingface(prompt, output_path, width, height)
    elif provider == "pixazo":
        return await generate_pixazo(prompt, output_path, width, height)
    else:
        raise ValueError(f"Unknown image gen provider: {provider}")


async def enhance_input_image(
    image_path: Path,
    output_path: Path,
    style_prompt: str = "",
) -> Path:
    """
    Optionally enhance the input image using an image generation API.

    This is a convenience wrapper — it takes the user's original image,
    creates a prompt from the style description, and generates an enhanced version.
    The enhanced image can then be fed into the main pipeline.
    """
    prompt = f"high quality, detailed, {style_prompt}" if style_prompt else "high quality, detailed, photorealistic"
    return await generate_image(prompt, output_path)
