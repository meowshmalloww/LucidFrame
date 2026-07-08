"""
Resource Monitor — 20% Safety Margins for VRAM, RAM, CPU.

Checks all resources before each pipeline stage. If any exceeds its
safe threshold (80% of total), the pipeline pauses with a WebSocket
warning and proceeds with reduced settings.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import psutil

logger = logging.getLogger(__name__)
TAG = "[RESOURCE]"

SAFE_MARGIN = float(os.getenv("VRAM_SAFETY_MARGIN", "0.20"))
RAM_MARGIN = float(os.getenv("RAM_SAFETY_MARGIN", "0.20"))
CPU_MARGIN = float(os.getenv("CPU_SAFETY_MARGIN", "0.20"))


@dataclass
class ResourceStatus:
    gpu: dict[str, Any] = field(default_factory=dict)
    ram: dict[str, Any] = field(default_factory=dict)
    cpu: dict[str, Any] = field(default_factory=dict)
    all_safe: bool = True
    warnings: list[str] = field(default_factory=list)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def get_vram_status() -> dict[str, Any]:
    """Get GPU VRAM usage. Returns empty dict if no CUDA."""
    if not _has_torch():
        return {"available": False}

    import torch
    if not torch.cuda.is_available():
        return {"available": False, "reason": "No CUDA device"}

    try:
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        budget = int(total * (1 - SAFE_MARGIN))

        return {
            "available": True,
            "name": props.name,
            "total_mb": total // (1024 * 1024),
            "allocated_mb": allocated // (1024 * 1024),
            "reserved_mb": reserved // (1024 * 1024),
            "free_mb": (total - allocated) // (1024 * 1024),
            "safe_budget_mb": budget // (1024 * 1024),
            "safe": allocated < budget,
            "usage_percent": (allocated / total) * 100 if total > 0 else 0,
        }
    except Exception as exc:
        logger.warning("%s Failed to get VRAM status: %s", TAG, exc)
        return {"available": False, "error": str(exc)}


def get_ram_status() -> dict[str, Any]:
    """Get system RAM usage."""
    mem = psutil.virtual_memory()
    safe_threshold = mem.total * (1 - RAM_MARGIN)
    return {
        "total_gb": round(mem.total / 1e9, 2),
        "used_gb": round(mem.used / 1e9, 2),
        "free_gb": round(mem.available / 1e9, 2),
        "usage_percent": mem.percent,
        "safe": mem.used < safe_threshold,
    }


def get_cpu_status() -> dict[str, Any]:
    """Get CPU usage."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    safe_threshold = 100 * (1 - CPU_MARGIN)
    return {
        "percent": cpu_percent,
        "core_count": psutil.cpu_count(),
        "safe": cpu_percent < safe_threshold,
    }


def check_all() -> ResourceStatus:
    """Check all resources. Returns ResourceStatus with warnings."""
    status = ResourceStatus()
    warnings: list[str] = []

    gpu = get_vram_status()
    status.gpu = gpu
    if gpu.get("available") and not gpu.get("safe", True):
        warnings.append(
            f"VRAM at {gpu['usage_percent']:.1f}% "
            f"({gpu['allocated_mb']}MB / {gpu['safe_budget_mb']}MB budget)"
        )
        status.all_safe = False

    ram = get_ram_status()
    status.ram = ram
    if not ram["safe"]:
        warnings.append(
            f"RAM at {ram['usage_percent']:.1f}% "
            f"({ram['used_gb']}GB / {ram['total_gb']}GB)"
        )
        status.all_safe = False

    cpu = get_cpu_status()
    status.cpu = cpu
    if not cpu["safe"]:
        warnings.append(
            f"CPU at {cpu['percent']:.1f}% (threshold 80%)"
        )
        status.all_safe = False

    status.warnings = warnings
    if warnings:
        for w in warnings:
            logger.warning("%s %s", TAG, w)
    else:
        logger.debug("%s All resources safe", TAG)

    return status


def get_health() -> dict[str, Any]:
    """Get full health snapshot for /health endpoint."""
    s = check_all()
    return {
        "status": "ok" if s.all_safe else "warning",
        "gpu": s.gpu,
        "ram": s.ram,
        "cpu": s.cpu,
        "warnings": s.warnings,
    }
