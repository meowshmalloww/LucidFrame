"""
Stage 2: Multi-View Hallucination via Zero123++.

Generates 6 consistent novel views from a single input image using
the Zero123++ diffusion model. All views are generated in a single
diffusion pass with shared noise and attention for temporal consistency.

Adapted from ParallaxVision's zero123_synth.py.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
TAG = "[MULTIVIEW]"

MULTIVIEW_MODEL = os.getenv("MULTIVIEW_MODEL", "zero123pp").lower()
MULTIVIEW_STEPS = int(os.getenv("MULTIVIEW_STEPS", "75"))

_PIPELINE: Any = None


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_zero123pp():
    """Load Zero123++ pipeline (lazy, cached)."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from diffusers import DiffusionPipeline

    logger.info("%s Loading Zero123++ v1.2 pipeline...", TAG)
    t0 = time.time()

    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    _PIPELINE = pipe
    logger.info("%s Zero123++ loaded in %.1fs", TAG, time.time() - t0)
    return _PIPELINE


def _split_grid(image_np: np.ndarray, rows: int = 3, cols: int = 2) -> list[np.ndarray]:
    """Split a 3x2 grid image into individual view images."""
    grid_h, grid_w = image_np.shape[:2]
    cell_h = grid_h // rows
    cell_w = grid_w // cols

    views = []
    for row in range(rows):
        for col in range(cols):
            cell = image_np[
                row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w,
            ]
            views.append(cell)

    return views


def generate_multiview(
    image_path: Path,
    output_dir: Path,
    target_size: int = 256,
) -> dict[str, Any]:
    """
    Generate 6 multi-view images from a single input image using Zero123++.

    Args:
        image_path: Path to the input image.
        output_dir: Directory to save generated views.
        target_size: Input resize dimension (256 or 320).

    Returns:
        Dict with:
            - views: list of np.ndarray (RGB, original resolution)
            - view_paths: list of Path to saved images
            - generation_time_sec: float
            - errors: list of error strings
    """
    t0 = time.time()
    errors: list[str] = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _has_cuda():
        return {
            "views": [],
            "view_paths": [],
            "generation_time_sec": 0.0,
            "errors": ["CUDA not available — cannot run Zero123++"],
        }

    import torch
    from PIL import Image

    try:
        pipe = _load_zero123pp()

        # Load and prepare input image
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_rgb.shape[:2]

        # Resize to target for Zero123++
        image_resized = cv2.resize(image_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(image_resized)

        logger.info("%s Generating 6 views from %dx%d image...", TAG, w_orig, h_orig)

        with torch.no_grad():
            result = pipe(pil_img, num_inference_steps=MULTIVIEW_STEPS)

        # Output is a 3x2 grid
        output_np = np.array(result.images[0])
        grid_views = _split_grid(output_np, rows=3, cols=2)

        # Resize each view back to original resolution
        views = []
        view_paths = []
        for i, view in enumerate(grid_views):
            view_resized = cv2.resize(view, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)
            views.append(view_resized)

            fname = f"view_{i:02d}.png"
            save_path = output_dir / fname
            cv2.imwrite(str(save_path), cv2.cvtColor(view_resized, cv2.COLOR_RGB2BGR))
            view_paths.append(save_path)

        gen_time = time.time() - t0
        logger.info("%s Generated %d views in %.1fs", TAG, len(views), gen_time)

        return {
            "views": views,
            "view_paths": [str(p) for p in view_paths],
            "generation_time_sec": gen_time,
            "errors": errors,
        }

    except Exception as exc:
        import traceback
        logger.error("%s Multi-view generation failed: %s\n%s", TAG, exc, traceback.format_exc())
        errors.append(str(exc))
        return {
            "views": [],
            "view_paths": [],
            "generation_time_sec": time.time() - t0,
            "errors": errors,
        }


def unload():
    """Unload the Zero123++ pipeline and free VRAM."""
    global _PIPELINE
    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("%s Zero123++ pipeline unloaded", TAG)
