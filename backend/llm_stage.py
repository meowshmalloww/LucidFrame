"""
Stage 1b: LLM — Dream Narrative Generation via Cloud API.

Takes the VLM's structured visual analysis and generates a dreamlike
"Master Scene Prompt" imagining what's *beyond* the frame.

The LLM is the "imagination" — it dreams what's beyond the image.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
TAG = "[LLM]"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline").lower()

LLM_SYSTEM_PROMPT = """You are the dream engine of LucidFrame, an AI art installation that hallucinates walkable 3D worlds from single images.

Your job: take a visual analysis of an image and imagine what lies BEYOND the frame. Not what's literally there — but what a lucid dream of this image would reveal.

Write a "Master Scene Prompt" — 2-3 sentences that describe the dreamlike space surrounding the image. This prompt will be used to condition a diffusion model that generates unseen views.

Rules:
- Be dreamlike and evocative, not literal or realistic
- Expand the atmosphere, not just the geometry
- Include sensory details: light, texture, color, mood
- The space should feel like you stepped THROUGH the image into a dream
- Keep it concise: 2-3 sentences maximum
- Do not use quotes or markdown formatting

Examples:
- "A fog-drenched corridor of weathered stone arches extends into amber twilight, where golden particles drift through shafts of dying light. The walls breathe with moss-covered textures, and the floor dissolves into a mirror of dark water reflecting an impossible sky."

- "Neon-lit rain falls upward in a crystalline cavern of fractured mirrors, each reflecting a different memory of a city that never existed. The ground pulses with soft bioluminescent veins, and the air tastes of ozone and old photographs."

Now generate a Master Scene Prompt for this visual analysis:"""


async def generate_dream_prompt_openai(analysis: dict[str, Any], api_key: str) -> str:
    """Call GPT-4o to generate dreamlike narrative from VLM analysis."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    import json
    analysis_text = json.dumps(analysis, indent=2)

    logger.info("%s Calling GPT-4o for dream narrative...", TAG)
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Visual analysis:\n{analysis_text}\n\nGenerate the Master Scene Prompt:"},
        ],
        max_tokens=300,
        temperature=0.9,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_anthropic(analysis: dict[str, Any], api_key: str) -> str:
    """Call Claude to generate dreamlike narrative from VLM analysis."""
    import anthropic
    import json

    client = anthropic.AsyncAnthropic(api_key=api_key)
    analysis_text = json.dumps(analysis, indent=2)

    logger.info("%s Calling Claude for dream narrative...", TAG)
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=LLM_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Visual analysis:\n{analysis_text}\n\nGenerate the Master Scene Prompt:",
            }
        ],
    )

    prompt = response.content[0].text.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_gemini(analysis: dict[str, Any], api_key: str) -> str:
    """Call Gemini to generate dreamlike narrative from VLM analysis."""
    import google.generativeai as genai
    import json

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    analysis_text = json.dumps(analysis, indent=2)

    logger.info("%s Calling Gemini for dream narrative...", TAG)
    response = await model.generate_content_async(
        f"{LLM_SYSTEM_PROMPT}\n\nVisual analysis:\n{analysis_text}\n\nGenerate the Master Scene Prompt:"
    )

    prompt = response.text.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_groq(analysis: dict[str, Any], api_key: str) -> str:
    """Call Groq to generate dreamlike narrative from VLM analysis."""
    from openai import AsyncOpenAI
    import json

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    analysis_text = json.dumps(analysis, indent=2)
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    logger.info("%s Calling Groq for dream narrative (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Visual analysis:\n{analysis_text}\n\nGenerate the Master Scene Prompt:"},
        ],
        max_tokens=300,
        temperature=0.9,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_openrouter(analysis: dict[str, Any], api_key: str) -> str:
    """Call OpenRouter to generate dreamlike narrative from VLM analysis."""
    from openai import AsyncOpenAI
    import json

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    analysis_text = json.dumps(analysis, indent=2)
    model = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-exp:free")

    logger.info("%s Calling OpenRouter for dream narrative (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Visual analysis:\n{analysis_text}\n\nGenerate the Master Scene Prompt:"},
        ],
        max_tokens=300,
        temperature=0.9,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_cerebras(analysis: dict[str, Any], api_key: str) -> str:
    """Call Cerebras for text generation — ultra-fast, text-only (no vision)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.cerebras.ai/v1",
    )
    model = os.getenv("LLM_MODEL", "llama-3.3-70b")

    logger.info("%s Calling Cerebras LLM API (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(analysis, indent=2)},
        ],
        max_tokens=300,
        temperature=0.8,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_nim(analysis: dict[str, Any], api_key: str) -> str:
    """Call NVIDIA NIM for text generation."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )
    model = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")

    logger.info("%s Calling NVIDIA NIM LLM API (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(analysis, indent=2)},
        ],
        max_tokens=300,
        temperature=0.8,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt_mistral(analysis: dict[str, Any], api_key: str) -> str:
    """Call Mistral for text generation."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
    )
    model = os.getenv("LLM_MODEL", "mistral-large-latest")

    logger.info("%s Calling Mistral LLM API (model=%s)...", TAG, model)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(analysis, indent=2)},
        ],
        max_tokens=300,
        temperature=0.8,
    )

    prompt = response.choices[0].message.content.strip()
    logger.info("%s Master Scene Prompt generated (%d chars): %s", TAG, len(prompt), prompt[:100])
    return prompt


async def generate_dream_prompt(analysis: dict[str, Any]) -> str:
    """
    Main entry point: generate a dreamlike Master Scene Prompt from VLM analysis.

    Args:
        analysis: Structured visual analysis dict from VLM stage.

    Returns:
        Master Scene Prompt string (2-3 sentences of dreamlike scene description).
    Defaults to an offline composition from the local visual analysis.
    """
    provider = LLM_PROVIDER

    if provider in {"offline", "local", "mock"}:
        return _mock_dream_prompt(analysis)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("%s OPENAI_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_openai(analysis, api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("%s ANTHROPIC_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_anthropic(analysis, api_key)

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("%s GEMINI_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_gemini(analysis, api_key)

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("%s GROQ_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_groq(analysis, api_key)

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            logger.warning("%s OPENROUTER_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_openrouter(analysis, api_key)

    elif provider == "cerebras":
        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if not api_key:
            logger.warning("%s CEREBRAS_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_cerebras(analysis, api_key)

    elif provider == "nim":
        api_key = os.getenv("NIM_API_KEY", "")
        if not api_key:
            logger.warning("%s NIM_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_nim(analysis, api_key)

    elif provider == "mistral":
        api_key = os.getenv("MISTRAL_API_KEY", "")
        if not api_key:
            logger.warning("%s MISTRAL_API_KEY not set — using mock dream prompt", TAG)
            return _mock_dream_prompt(analysis)
        return await generate_dream_prompt_mistral(analysis, api_key)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def _mock_dream_prompt(analysis: dict[str, Any]) -> str:
    """Generate a mock dream prompt from analysis keywords for offline testing."""
    mood = analysis.get("mood", "ethereal")
    atmosphere = analysis.get("atmosphere", "dreamy")
    lighting = analysis.get("lighting", "soft diffused")
    style = analysis.get("style", "photograph")
    return (
        f"A {mood} dreamscape unfolds beyond the frame, bathed in {lighting} that "
        f"filters through an {atmosphere} space. The {style} world extends into "
        f"depths of impossible geometry, where every surface whispers of memories "
        f"not yet lived. Shadows dance with light as the boundary between real and "
        f"dreamed dissolves into pure sensation."
    )
