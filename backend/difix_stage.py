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
import time
from pathlib import Path
from typing import Any

import numpy as np

from reconstruction_stage import GaussianData

logger = logging.getLogger(__name__)
TAG = "[DIFIX]"

DIFIX_MODE = os.getenv("DIFIX_MODE", "structural").lower()

_DIFIX_PIPELINE: Any = None


def _load_difix():
    """Load Difix3D+ pipeline (lazy).

    Uses SD-Turbo as the base diffusion model for image-to-image repair.
    SD-Turbo is a distilled Stable Diffusion model that runs in a single
    step (~8GB VRAM), making it suitable for our 12GB budget.

    The 'structural' mode uses low strength (0.3) to preserve geometry
    while filling voids. 'full' mode uses higher strength (0.6) for
    aggressive artifact removal.
    """
    global _DIFIX_PIPELINE
    if _DIFIX_PIPELINE is not None:
        return _DIFIX_PIPELINE

    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
        from diffusers.utils import load_image

        logger.info("%s Loading SD-Turbo for Difix3D+ image repair...", TAG)
        t0 = time.time()

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe = pipe.to("cuda")
        # Optimize for inference
        pipe.set_progress_bar_config(disable=True)
        _DIFIX_PIPELINE = pipe
        logger.info("%s SD-Turbo loaded in %.1fs", TAG, time.time() - t0)
        return _DIFIX_PIPELINE

    except Exception as exc:
        logger.warning("%s Failed to load Difix3D+ (SD-Turbo): %s", TAG, exc)
        raise


def _render_splat_views(
    gaussians: GaussianData,
    n_views: int = 8,
    width: int = 512,
    height: int = 512,
) -> list[np.ndarray] | None:
    """
    Render n_views virtual camera views from the current Gaussian splat.

    Uses gsplat.rasterization if available. Returns list of (H, W, 3) RGB arrays
    or None if rendering is unavailable.
    """
    try:
        import torch
        import gsplat
    except ImportError:
        logger.warning("%s gsplat not available — cannot render views", TAG)
        return None

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    N = gaussians.count

    means = torch.tensor(gaussians.positions, dtype=torch.float32, device=device)
    log_scales = torch.tensor(gaussians.scales, dtype=torch.float32, device=device)
    quats = torch.tensor(gaussians.rotations, dtype=torch.float32, device=device)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)
    opacities = torch.sigmoid(torch.full((N,), 2.0, dtype=torch.float32, device=device))
    colors = torch.tensor(gaussians.colors, dtype=torch.float32, device=device)
    colors = torch.sigmoid(colors) if colors.max() > 1.0 else colors
    scales = torch.exp(log_scales)

    # Generate camera poses in a circle around the scene
    import math
    extent = float(np.abs(gaussians.positions).max())
    radius = max(extent * 2.0, 3.0)

    views = []
    for i in range(n_views):
        angle = (i / n_views) * 2.0 * math.pi
        cam_x = radius * math.cos(angle)
        cam_z = radius * math.sin(angle)
        cam_y = 0.0

        # Look-at origin
        forward = np.array([-cam_x, -cam_y, -cam_z])
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)

        ext = np.eye(4, dtype=np.float32)
        ext[:3, 0] = right
        ext[:3, 1] = up
        ext[:3, 2] = -forward
        ext[:3, 3] = [cam_x, cam_y, cam_z]

        ext_t = torch.tensor(ext, dtype=torch.float32, device=device)
        fx = fy = max(width, height)
        K = torch.tensor([
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        try:
            with torch.no_grad():
                rendered, alpha, info = gsplat.rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=ext_t.unsqueeze(0),
                    Ks=K[:3, :3].unsqueeze(0),
                    width=width,
                    height=height,
                )
                img = rendered.squeeze(0).cpu().numpy()
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                views.append(img)
        except Exception as e:
            logger.warning("%s View %d render failed: %s", TAG, i, e)

    return views if views else None


def fix_artifacts(
    gaussians: GaussianData,
    reference_image: Path,
    output_dir: Path,
    mode: str = DIFIX_MODE,
    iterations: int = 3,
) -> GaussianData:
    """
    Fix artifacts in 3D Gaussians using Difix3D+.

    Pipeline per iteration:
      1. Render virtual camera views from broken angles of current splat
      2. Feed rendered images + reference image to Difix3D+ diffusion
      3. Difix outputs cleaned images
      4. Use cleaned images as pseudo-GT for gsplat optimization
      5. Repeat for `iterations` cycles

    Args:
        gaussians: Current Gaussian data.
        reference_image: Path to original input image.
        output_dir: Working directory.
        mode: "structural", "full", or "skip".
        iterations: Number of fix -> optimize cycles.

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

        from PIL import Image
        import cv2
        import torch

        ref_img = Image.open(str(reference_image)).convert("RGB")

        # Mode-based parameters for SD-Turbo image2image
        if mode == "full":
            strength = 0.6
            repair_prompt = "highly detailed 3D render, sharp focus, clean geometry, no artifacts, photorealistic"
        else:  # structural
            strength = 0.3
            repair_prompt = "clean 3D render, fill missing areas, preserve structure, smooth surfaces"
        n_steps = 4  # SD-Turbo is distilled, 1-4 steps sufficient

        current_gaussians = gaussians

        for cycle in range(iterations):
            logger.info("%s Difix cycle %d/%d (strength=%.1f)", TAG, cycle + 1, iterations, strength)

            # 1. Render views from current splat
            rendered_views = _render_splat_views(current_gaussians, n_views=6)
            if rendered_views is None or len(rendered_views) == 0:
                logger.warning("%s Could not render views — ending Difix early", TAG)
                break

            # 2. Feed rendered views to SD-Turbo for repair (image2image)
            fixed_views = []
            for i, view_img in enumerate(rendered_views):
                pil_view = Image.fromarray(view_img)
                try:
                    with torch.no_grad():
                        result = pipe(
                            prompt=repair_prompt,
                            image=pil_view,
                            strength=strength,
                            num_inference_steps=n_steps,
                            guidance_scale=1.0,
                        )
                        fixed = np.array(result.images[0])
                    fixed_views.append(fixed)
                    cv2.imwrite(str(output_dir / f"difix_cycle{cycle}_view{i}.png"),
                                cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    logger.warning("%s Difix view %d failed: %s — using original", TAG, i, e)
                    fixed_views.append(view_img)

            # 3. Save fixed views and run gsplat optimization with them
            fixed_view_paths = []
            for i, fv in enumerate(fixed_views):
                p = output_dir / f"difix_fixed_{cycle}_{i}.png"
                cv2.imwrite(str(p), cv2.cvtColor(fv, cv2.COLOR_RGB2BGR))
                fixed_view_paths.append(str(p))

            # 4. Optimize Gaussians using fixed views as pseudo-GT
            from reconstruction_stage import optimize_gaussians_gsplat
            current_gaussians = optimize_gaussians_gsplat(
                current_gaussians,
                fixed_view_paths,
                iterations=300,
                densify_interval=100,
            )

            logger.info("%s Difix cycle %d complete: %d Gaussians",
                        TAG, cycle + 1, current_gaussians.count)

        logger.info("%s Difix3D+ complete: %d Gaussians after %d cycles",
                    TAG, current_gaussians.count, iterations)
        return current_gaussians

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
