"""
Progress Broadcaster — WebSocket event streaming for pipeline stages.

Each stage sends: stage_start, stage_progress, stage_done events.
The frontend DreamLog component displays these in real-time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)
TAG = "[PROGRESS]"


@dataclass
class ProgressBroadcaster:
    """Broadcasts pipeline events to a WebSocket connection."""
    send_fn: Callable[[dict], Any]
    job_id: str = ""
    _stage_timings: dict[str, float] = field(default_factory=dict)

    async def _send(self, event: dict[str, Any]):
        event["timestamp"] = time.time()
        event["job_id"] = self.job_id
        try:
            await self.send_fn(event)
        except Exception as exc:
            logger.warning("%s Failed to send event: %s", TAG, exc)

    async def stage_start(self, stage: str, message: str = ""):
        self._stage_timings[stage] = time.time()
        await self._send({
            "event": "stage_start",
            "stage": stage,
            "message": message,
        })

    async def stage_progress(self, stage: str, message: str, data: dict | None = None):
        event: dict[str, Any] = {
            "event": "stage_progress",
            "stage": stage,
            "message": message,
        }
        if data:
            event["data"] = data
        await self._send(event)

    async def stage_done(self, stage: str, data: dict | None = None):
        elapsed = time.time() - self._stage_timings.get(stage, time.time())
        event: dict[str, Any] = {
            "event": "stage_done",
            "stage": stage,
            "elapsed_sec": round(elapsed, 2),
        }
        if data:
            event["data"] = data
        await self._send(event)
        logger.info("%s Stage '%s' done in %.1fs", TAG, stage, elapsed)

    async def pipeline_done(self, splat_url: str):
        await self._send({
            "event": "pipeline_done",
            "splat_url": splat_url,
        })

    async def error(self, stage: str, error_msg: str):
        await self._send({
            "event": "error",
            "stage": stage,
            "error": error_msg,
        })

    async def warning(self, stage: str, message: str):
        await self._send({
            "event": "warning",
            "stage": stage,
            "message": message,
        })
