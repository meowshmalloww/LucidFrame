"""
VRAM Manager — Sequential model loading with explicit cleanup.

Only one heavy model in VRAM at a time. Between stages, we call
del + torch.cuda.empty_cache() to free memory.
"""
from __future__ import annotations

import gc
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)
TAG = "[VRAM]"


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def cleanup():
    """Free all unused GPU memory."""
    gc.collect()
    if _has_torch():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("%s cleanup done, allocated: %dMB",
                        TAG, torch.cuda.memory_allocated(0) // (1024 * 1024))


def load_model(load_fn, *args, **kwargs) -> Any:
    """Load a model with VRAM tracking. Calls cleanup before loading."""
    if not _has_torch():
        raise RuntimeError("PyTorch not available")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — cannot load GPU model")

    cleanup()

    t0 = time.time()
    before_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
    logger.info("%s Loading model (before: %dMB allocated)...", TAG, before_mb)

    model = load_fn(*args, **kwargs)

    after_mb = torch.cuda.memory_allocated(0) // (1024 * 1024)
    delta = after_mb - before_mb
    logger.info("%s Model loaded in %.1fs (delta: +%dMB, total: %dMB)",
                TAG, time.time() - t0, delta, after_mb)

    return model


def unload_model(model: Any):
    """Unload a model and free VRAM."""
    if model is None:
        return

    logger.info("%s Unloading model...", TAG)
    del model
    cleanup()
    logger.info("%s Model unloaded", TAG)


def get_allocated_mb() -> int:
    """Get current VRAM allocation in MB."""
    if not _has_torch():
        return 0
    import torch
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated(0) // (1024 * 1024)
