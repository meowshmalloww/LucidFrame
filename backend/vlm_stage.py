"""
Stage 1a: VLM — Image Understanding via Cloud API.

Sends the input image to GPT-4o Vision API and receives structured
visual analysis (era, style, mood, lighting, architecture, etc.).

The VLM is the "eyes" — it extracts what's *in* the image.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)
TAG = "[VLM]"

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "openai").lower()

VLM_ANALYSIS_PROMPT = """You are a visual analyst examining an image for an AI art project called LucidFrame.
Analyze this image and return a JSON object with the following fields:

{
  "era": "What time period does this image evoke? (e.g., '1920s Art Deco', 'medieval', '1970s polaroid', 'timeless')",
  "style": "Artistic style (e.g., 'oil painting', 'photograph', 'digital art', 'ink wash', 'vintage photograph')",
  "mood": "Emotional tone (e.g., 'melancholic', 'ethereal', 'ominous', 'nostalgic', 'serene')",
  "lighting": "Lighting description (e.g., 'golden hour', 'harsh shadows', 'soft diffused', 'neon glow')",
  "architecture": "Architectural or structural elements visible (e.g., 'gothic arches', 'modern glass', 'none', 'ruins')",
  "color_palette": ["list", "of", "dominant", "colors", "as", "hex", "or", "names"],
  "atmosphere": "Overall atmosphere (e.g., 'foggy and mysterious', 'bright and airy', 'dark and claustrophobic')",
  "visible_content": "Brief description of what is visible in the image (2-3 sentences)",
  "textures": ["list", "of", "dominant", "textures", "e.g.", 'rough stone', 'smooth glass', 'woven fabric']",
  "composition": "Composition notes (e.g., 'centered subject', 'rule of thirds', 'symmetrical', 'wide angle')"
}

Be creative and evocative. This analysis will be used to generate a dreamlike 3D world.
Return ONLY the JSON object, no markdown or explanation."""


def _encode_image_b64(image_path: Path) -> str:
    """Read image file and encode as base64."""
    with open(image_path, "rb") as f:
        data = f.read()
    ext = image_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"


async def analyze_image_openai(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call GPT-4o Vision API for image analysis."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    image_b64 = _encode_image_b64(image_path)

    logger.info("%s Calling GPT-4o Vision API...", TAG)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VLM_ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_b64}},
                ],
            }
        ],
        max_tokens=1000,
        temperature=0.7,
    )

    content = response.choices[0].message.content
    logger.info("%s VLM response received (%d chars)", TAG, len(content))

    try:
        analysis = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            analysis = json.loads(content[start:end])
        else:
            analysis = {"raw_response": content}

    return analysis


async def analyze_image_anthropic(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call Claude Vision API for image analysis."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    image_b64 = _encode_image_b64(image_path)
    ext = image_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"

    logger.info("%s Calling Claude Vision API...", TAG)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{ext}",
                            "data": image_b64.split(",")[1],
                        },
                    },
                    {"type": "text", "text": VLM_ANALYSIS_PROMPT},
                ],
            }
        ],
    )

    content = response.content[0].text
    logger.info("%s VLM response received (%d chars)", TAG, len(content))

    try:
        analysis = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            analysis = json.loads(content[start:end])
        else:
            analysis = {"raw_response": content}

    return analysis


async def analyze_image(image_path: Path) -> dict[str, Any]:
    """
    Main entry point: analyze an image using the configured VLM provider.

    Returns structured visual analysis dict.
    """
    provider = VLM_PROVIDER

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return await analyze_image_openai(image_path, api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        return await analyze_image_anthropic(image_path, api_key)

    else:
        raise ValueError(f"Unknown VLM provider: {provider}")
