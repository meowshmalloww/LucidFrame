"""
Stage 1b: LLM — Dream Narrative Generation via Cloud API.

Takes the VLM's structured visual analysis and generates a dreamlike
"Master Scene Prompt" imagining what's *beyond* the frame.

The LLM is the "imagination" — it dreams what's beyond the image.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)
TAG = "[LLM]"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

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


async def generate_dream_prompt(analysis: dict[str, Any]) -> str:
    """
    Main entry point: generate a dreamlike Master Scene Prompt from VLM analysis.

    Args:
        analysis: Structured visual analysis dict from VLM stage.

    Returns:
        Master Scene Prompt string (2-3 sentences of dreamlike scene description).
    """
    provider = LLM_PROVIDER

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        return await generate_dream_prompt_openai(analysis, api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        return await generate_dream_prompt_anthropic(analysis, api_key)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
