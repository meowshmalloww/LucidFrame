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


async def analyze_image_gemini(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call Gemini Vision API for image analysis."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    img = Path(image_path).read_bytes()
    ext = image_path.suffix.lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"

    logger.info("%s Calling Gemini Vision API...", TAG)
    response = await model.generate_content_async(
        [VLM_ANALYSIS_PROMPT, {"mime_type": f"image/{ext}", "data": img}]
    )

    content = response.text
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


async def analyze_image_groq(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call Groq Vision API (Llama 3.2 Vision) for image analysis."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    image_b64 = _encode_image_b64(image_path)

    logger.info("%s Calling Groq Vision API (Llama 3.2 90B Vision)...", TAG)
    response = await client.chat.completions.create(
        model="meta-llama/llama-3.2-90b-vision-preview",
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


async def analyze_image_openrouter(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call OpenRouter Vision API for image analysis.

    OpenRouter routes to many models. Default uses google/gemini-2.0-flash-exp:free
    which has excellent vision and a free tier.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    image_b64 = _encode_image_b64(image_path)
    model = os.getenv("VLM_MODEL", "google/gemini-2.0-flash-exp:free")

    logger.info("%s Calling OpenRouter Vision API (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
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


async def analyze_image_nim(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call NVIDIA NIM Vision API for image analysis.

    NVIDIA NIM provides OpenAI-compatible endpoints for vision models
    like meta/llama-3.2-90b-vision-instruct and nvidia/neva-22b.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    image_b64 = _encode_image_b64(image_path)
    model = os.getenv("VLM_MODEL", "meta/llama-3.2-90b-vision-instruct")

    logger.info("%s Calling NVIDIA NIM Vision API (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
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


async def analyze_image_mistral(image_path: Path, api_key: str) -> dict[str, Any]:
    """Call Mistral Vision API for image analysis.

    Uses pixtral-12b-2409 model via OpenAI-compatible endpoint.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
    )
    image_b64 = _encode_image_b64(image_path)
    model = os.getenv("VLM_MODEL", "pixtral-12b-2409")

    logger.info("%s Calling Mistral Vision API (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
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


async def analyze_image(image_path: Path) -> dict[str, Any]:
    """
    Main entry point: analyze an image using the configured VLM provider.

    Supported providers: openai, anthropic, gemini, groq, openrouter
    Falls back to a mock analysis if no API key is configured.
    """
    provider = VLM_PROVIDER

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("%s OPENAI_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_openai(image_path, api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("%s ANTHROPIC_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_anthropic(image_path, api_key)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("%s GEMINI_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_gemini(image_path, api_key)

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("%s GROQ_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_groq(image_path, api_key)

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("%s OPENROUTER_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_openrouter(image_path, api_key)

    elif provider == "nim":
        api_key = os.getenv("NIM_API_KEY", "")
        if not api_key:
            logger.warning("%s NIM_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_nim(image_path, api_key)

    elif provider == "cerebras":
        logger.warning("%s Cerebras has no vision models — using mock analysis", TAG)
        return _mock_analysis()

    elif provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            logger.warning("%s MISTRAL_API_KEY not set — using mock analysis", TAG)
            return _mock_analysis()
        return await analyze_image_mistral(image_path, api_key)

    else:
        raise ValueError(f"Unknown VLM provider: {provider}")


def _mock_analysis() -> dict[str, Any]:
    """Return a generic creative analysis for offline/local testing."""
    return {
        "era": "timeless",
        "style": "photograph",
        "mood": "mysterious",
        "lighting": "soft diffused",
        "architecture": "none",
        "color_palette": ["#2a2a3e", "#6c5ce7", "#0d1117", "#e8e8f0"],
        "atmosphere": "dreamy and ethereal",
        "visible_content": "A captivating scene with rich textures and depth.",
        "textures": ["smooth", "rough", "reflective"],
        "composition": "centered subject with atmospheric background",
    }
