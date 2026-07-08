"""
Stage 4: Difix3D+ Artifact Fixing (Optional, Tunable).

Renders virtual camera views from "broken" angles of the current splat,
feeds them to Difix3D+ for repair, then uses the cleaned images as
pseudo-GT for gsplat optimization iterations.

Modes:
  - structural: Only fix black voids / missing geometry holes.
  - full: Aggressive artifact removal.
  - skip: Don't run Difix at all (raw output may be more artistic).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from reconstruction_stage import GaussianData

logger = logging.getLogger(__name__)
TAG = "[DIFIX]"

DIFIX_MODE = os.getenv("DIFIX_MODE", "structural").lower()

_DIFIX_PIPELINE: Any = None


def _load_difix():
    """Load Difix3D+ pipeline (lazy)."""
    global _DIFIX_PIPELINE
    if _DIFIX_PIPELINE is not None:
        return _DIFIX_PIPELINE

    try:
        import torch
        from diffusers import DiffusionPipeline

        logger.info("%s Loading Difix3D+ pipeline...", TAG)
        pipe = DiffusionPipeline.from_pretrained(
            "nvidia/difix",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to("cuda")
        _DIFIX_PIPELINE = pipe
        logger.info("%s Difix3D+ loaded", TAG)
        return _DIFIX_PIPELINE

    except Exception as exc:
        logger.warning("%s Failed to load Difix3D+: %s", TAG, exc)
        raise


def fix_artifacts(
    gaussians: GaussianData,
    reference_image: Path,
    output_dir: Path,
    mode: str = DIFIX_MODE,
    iterations: int = 3,
) -> GaussianData:
    """
    Fix artifacts in 3D Gaussians using Difix3D+.

    Args:
        gaussians: Current Gaussian data.
        reference_image: Path to original input image.
        output_dir: Working directory.
        mode: "structural", "full", or "skip".
        iterations: Number of fix → optimize cycles.

    Returns:
        Updated GaussianData with fixed artifacts.
    """
    if mode == "skip":
        logger.info("%s Difix skipped (skip mode)", TAG)
        return gaussians

    if gaussians.count == 0:
        logger.warning("%s No Gaussians to fix", TAG)
        return gaussians

    try:
        pipe = _load_difix()

        logger.info("%s Running Difix3D+ in '%s' mode (%d iterations)...", TAG, mode, iterations)

        # TODO: Full Difix3D+ integration:
        # 1. Render virtual camera views from broken angles of current splat
        # 2. Feed rendered images + reference image to Difix
        # 3. Difix outputs cleaned images
        # 4. Use cleaned images as pseudo-GT for gsplat optimization
        # 5. Repeat for `iterations` cycles

        # For now, return the input Gaussians unchanged
        # The actual implementation requires:
        # - A Gaussian rasterizer to render views from the splat
        # - The Difix pipeline to fix rendered images
        # - A gsplat optimization loop to refine Gaussians

        logger.warning("%s Difix3D+ full integration pending — returning unchanged Gaussians", TAG)
        return gaussians

    except Exception as exc:
        logger.error("%s Difix3D+ failed: %s — returning unchanged Gaussians", TAG, exc)
        return gaussians


def unload():
    """Unload Difix pipeline and free VRAM."""
    global _DIFIX_PIPELINE
    if _DIFIX_PIPELINE is not None:
        del _DIFIX_PIPELINE
        _DIFIX_PIPELINE = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("%s Difix3D+ pipeline unloaded", TAG)
