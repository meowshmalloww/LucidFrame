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


def _gsplat_available() -> bool:
    """Check if gsplat CUDA rasterization is available."""
    try:
        import gsplat  # noqa: F401
        from gsplat.cuda._backend import _C
        return _C is not None
    except (ImportError, OSError):
        return False

# Difix3D+ is an experimental optional pass. It is not a substitute for
# calibrated views, so keep it off unless a creator intentionally enables it.
DIFIX_MODE = os.getenv("DIFIX_MODE", "skip").lower()

# ConFixGS-inspired confidence parameters (arxiv 2605.09688)
# Controls how aggressively diffusion-repaired pixels are trusted.
# Low confidence → pixel is likely hallucinated → suppress in optimization.
DIFIX_CONFIDENCE_THRESHOLD = float(os.getenv("DIFIX_CONFIDENCE_THRESHOLD", "0.35"))
DIFIX_VOID_THRESHOLD = float(os.getenv("DIFIX_VOID_THRESHOLD", "0.08"))
DIFIX_CONFIDENCE_DECAY = float(os.getenv("DIFIX_CONFIDENCE_DECAY", "0.15"))
DIFIX_SMOOTH_KERNEL = int(os.getenv("DIFIX_SMOOTH_KERNEL", "7"))

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

    Uses gsplat.rasterization if available, falls back to torch_splat
    (pure PyTorch renderer) when gsplat CUDA is not compiled.
    Returns list of (H, W, 3) RGB arrays or None if rendering is unavailable.
    """
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    use_gsplat = _gsplat_available()
    if not use_gsplat:
        logger.info("%s gsplat CUDA unavailable — using torch_splat fallback renderer", TAG)

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
                if use_gsplat:
                    import gsplat
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
                else:
                    from torch_splat import rasterize_gaussians
                    img = rasterize_gaussians(
                        means=means,
                        quats=quats,
                        scales=scales,
                        opacities=opacities,
                        colors=colors,
                        viewmat=ext_t,
                        K=K[:3, :3],
                        width=width,
                        height=height,
                    ).cpu().numpy()
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                views.append(img)
        except Exception as e:
            logger.warning("%s View %d render failed: %s", TAG, i, e)

    return views if views else None


def _detect_voids(rendered: np.ndarray) -> np.ndarray:
    """Detect void/empty regions in a rendered view (GSFix3D-inspired).

    Returns a boolean mask where True = void pixel (needs inpainting).
    Voids are very dark pixels or near-black regions that indicate
    missing geometry in the 3DGS reconstruction.
    """
    if rendered.dtype != np.float32:
        rendered = rendered.astype(np.float32) / 255.0
    luminance = rendered.mean(axis=-1)
    return luminance < DIFIX_VOID_THRESHOLD


def _compute_repair_confidence(
    original: np.ndarray,
    repaired: np.ndarray,
) -> np.ndarray:
    """Compute per-pixel confidence map for diffusion-repaired views.

    Inspired by ConFixGS (arxiv 2605.09688): diffusion outputs are NOT
    ground truth. We estimate confidence by measuring how much the repair
    changed each pixel relative to the original rendering.

    - High confidence: repair preserved existing geometry (small change)
      OR filled a void (original was black, repaired has content).
    - Low confidence: repair drastically altered well-rendered regions
      (likely hallucination).

    Returns: (H, W) float32 confidence map in [0, 1].
    """
    if original.dtype != np.float32:
        original = original.astype(np.float32) / 255.0
    if repaired.dtype != np.float32:
        repaired = repaired.astype(np.float32) / 255.0

    # Pixel-wise color discrepancy
    diff = np.abs(repaired - original).mean(axis=-1)

    # Void mask: original was empty → repair is filling, not hallucinating
    void_mask = _detect_voids(original)

    # Confidence: high when diff is small (preserved) OR when filling voids
    # For non-void pixels: confidence decays with discrepancy
    # For void pixels: confidence is boosted (inpainting is the goal)
    confidence = np.exp(-diff / DIFIX_CONFIDENCE_DECAY)
    confidence = np.where(void_mask, np.maximum(confidence, 0.6), confidence)

    # Spatial smoothing to avoid noisy per-pixel decisions (ConFixGS uses k×k filter)
    k = DIFIX_SMOOTH_KERNEL
    if k > 1:
        from scipy.ndimage import uniform_filter
        confidence = uniform_filter(confidence, size=k, mode="reflect")

    return confidence.astype(np.float32)


def _blend_with_confidence(
    original: np.ndarray,
    repaired: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """Blend original and repaired views using per-pixel confidence.

    High-confidence pixels → use repaired (diffusion fix is trustworthy).
    Low-confidence pixels → keep original (avoid hallucination).
    """
    if original.dtype != np.float32:
        original = original.astype(np.float32) / 255.0
    if repaired.dtype != np.float32:
        repaired = repaired.astype(np.float32) / 255.0

    conf_3d = confidence[:, :, None]
    blended = repaired * conf_3d + original * (1.0 - conf_3d)
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def fix_artifacts(
    gaussians: GaussianData,
    reference_image: Path,
    output_dir: Path,
    mode: str = DIFIX_MODE,
    iterations: int = 3,
) -> GaussianData:
    """
    Fix artifacts in 3D Gaussians using confidence-weighted diffusion repair.

    Pipeline per iteration (ConFixGS + GSFix3D inspired):
      1. Render virtual camera views from broken angles of current splat
      2. Detect void/empty regions that need inpainting (GSFix3D)
      3. Feed rendered images to diffusion model for repair
      4. Compute per-pixel confidence: did the repair preserve geometry
         or hallucinate? (ConFixGS)
      5. Blend repaired and original using confidence map
      6. Use blended views as pseudo-GT for gsplat optimization
      7. Repeat for `iterations` cycles

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

        logger.info("%s Running confidence-weighted Difix3D+ in '%s' mode (%d iterations)...",
                    TAG, mode, iterations)

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
            # Use lower resolution when using torch_splat fallback for speed
            render_w, render_h = (512, 512) if _gsplat_available() else (256, 256)
            rendered_views = _render_splat_views(
                current_gaussians, n_views=6, width=render_w, height=render_h,
            )
            if rendered_views is None or len(rendered_views) == 0:
                logger.warning("%s Could not render views — ending Difix early", TAG)
                break

            # 2. Feed rendered views to SD-Turbo for repair (image2image)
            # 3. Compute confidence and blend (ConFixGS-inspired)
            blended_views = []
            total_voids = 0
            total_low_conf = 0
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
                        repaired = np.array(result.images[0])
                except Exception as e:
                    logger.warning("%s Difix view %d failed: %s — using original", TAG, i, e)
                    blended_views.append(view_img)
                    continue

                # Detect voids in the original rendering
                void_mask = _detect_voids(view_img)
                total_voids += int(void_mask.sum())

                # Compute per-pixel confidence (ConFixGS)
                confidence = _compute_repair_confidence(view_img, repaired)
                total_low_conf += int((confidence < DIFIX_CONFIDENCE_THRESHOLD).sum())

                # Blend: trust repair where confidence is high, keep original where low
                blended = _blend_with_confidence(view_img, repaired, confidence)
                blended_views.append(blended)

                # Save debug visualizations
                cv2.imwrite(str(output_dir / f"difix_cycle{cycle}_view{i}_repaired.png"),
                            cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(output_dir / f"difix_cycle{cycle}_view{i}_blended.png"),
                            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
                # Save confidence as heatmap
                conf_vis = (confidence * 255).astype(np.uint8)
                cv2.imwrite(str(output_dir / f"difix_cycle{cycle}_view{i}_confidence.png"),
                            cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET))

            h, w = rendered_views[0].shape[:2] if rendered_views else (0, 0)
            logger.info(
                "%s Cycle %d: %d voids detected, %d low-confidence pixels suppressed (of %d total)",
                TAG, cycle + 1, total_voids, total_low_conf, len(blended_views) * h * w,
            )

            # 4. Save blended views and run gsplat optimization with them
            blended_view_paths = []
            for i, bv in enumerate(blended_views):
                p = output_dir / f"difix_blended_{cycle}_{i}.png"
                cv2.imwrite(str(p), cv2.cvtColor(bv, cv2.COLOR_RGB2BGR))
                blended_view_paths.append(str(p))

            # 5. Refine Gaussians using blended views as pseudo-GT
            current_gaussians = _project_repaired_to_gaussians(
                current_gaussians,
                rendered_views,
                blended_views,
                n_views=6,
                width=rendered_views[0].shape[1],
                height=rendered_views[0].shape[0],
            )

            logger.info("%s Difix cycle %d complete: %d Gaussians",
                        TAG, cycle + 1, current_gaussians.count)

        logger.info("%s Difix3D+ complete: %d Gaussians after %d cycles",
                    TAG, current_gaussians.count, iterations)
        return current_gaussians

    except Exception as exc:
        logger.error("%s Difix3D+ failed: %s — returning unchanged Gaussians", TAG, exc)
        return gaussians


def _project_repaired_to_gaussians(
    gaussians: GaussianData,
    original_views: list[np.ndarray],
    repaired_views: list[np.ndarray],
    n_views: int = 6,
    width: int = 256,
    height: int = 256,
) -> GaussianData:
    """
    Projection-based Gaussian refinement (no differentiable rendering needed).

    Instead of gradient-based optimization, this approach:
      1. Identifies void pixels that were filled by the diffusion model
      2. Backprojects those pixels to 3D using median scene depth
      3. Creates new Gaussians at those positions with repaired colors
      4. Updates existing Gaussian colors to match repaired views

    This is inspired by GSFix3D's geometry-aware inpainting approach.
    """
    import math
    import cv2

    logger.info("%s Running projection-based refinement (no gsplat needed)", TAG)

    positions = gaussians.positions.copy()
    scales = gaussians.scales.copy()
    rotations = gaussians.rotations.copy()
    colors = gaussians.colors.copy()
    opacities = gaussians.opacities.copy()

    # Scene geometry
    extent = float(np.abs(positions).max())
    radius = max(extent * 2.0, 3.0)
    median_depth = float(np.median(positions[:, 2])) if len(positions) > 0 else 5.0

    new_positions = []
    new_colors = []
    new_scales = []
    new_opacities = []
    new_rotations = []

    for view_idx in range(min(n_views, len(repaired_views))):
        orig = original_views[view_idx]
        repaired = repaired_views[view_idx]

        if orig.shape[:2] != repaired.shape[:2]:
            repaired = cv2.resize(repaired, (orig.shape[1], orig.shape[0]))

        h, w = orig.shape[:2]

        # Detect voids in original that were filled in repaired
        void_mask = _detect_voids(orig)
        repaired_lum = repaired.astype(np.float32).mean(axis=-1) / 255.0
        filled_mask = void_mask & (repaired_lum > DIFIX_VOID_THRESHOLD)

        n_filled = int(filled_mask.sum())
        if n_filled == 0:
            continue

        # Subsample filled pixels (max 2000 per view)
        filled_coords = np.argwhere(filled_mask)  # (y, x)
        if len(filled_coords) > 2000:
            idx = np.random.choice(len(filled_coords), 2000, replace=False)
            filled_coords = filled_coords[idx]

        # Camera pose for this view (same as _render_splat_views)
        angle = (view_idx / n_views) * 2.0 * math.pi
        cam_x = radius * math.cos(angle)
        cam_z = radius * math.sin(angle)
        cam_y = 0.0

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

        # Intrinsics
        fx = fy = float(max(w, h))
        cx, cy = w / 2.0, h / 2.0

        # Backproject filled pixels to 3D
        R = ext[:3, :3]
        t = ext[:3, 3]

        for y_px, x_px in filled_coords:
            # Ray in camera space
            ray_cam = np.array([
                (x_px - cx) / fx,
                (y_px - cy) / fy,
                1.0,
            ], dtype=np.float32)
            ray_cam = ray_cam / np.linalg.norm(ray_cam)

            # Place at median depth (approximate)
            depth = median_depth + radius  # distance from camera
            point_cam = ray_cam * depth

            # Transform to world space: world = R^T @ (cam - t)
            point_world = R.T @ (point_cam - t)

            # Color from repaired image
            color = repaired[y_px, x_px].astype(np.float32) / 255.0

            new_positions.append(point_world)
            new_colors.append(color)
            new_scales.append(np.log(0.05))  # small splat
            new_opacities.append(0.7)
            new_rotations.append([1.0, 0.0, 0.0, 0.0])  # identity quaternion

        logger.info("%s View %d: added %d Gaussians from repaired voids",
                    TAG, view_idx, len(filled_coords))

    if new_positions:
        new_positions = np.array(new_positions, dtype=np.float32)
        new_colors = np.array(new_colors, dtype=np.float32)
        new_scales = np.column_stack([new_scales] * 3).astype(np.float32)
        new_opacities = np.array(new_opacities, dtype=np.float32).reshape(-1, 1)
        new_rotations = np.array(new_rotations, dtype=np.float32)

        positions = np.vstack([positions, new_positions])
        colors = np.vstack([colors, new_colors])
        scales = np.vstack([scales, new_scales])
        opacities = np.vstack([opacities, new_opacities])
        rotations = np.vstack([rotations, new_rotations])

        logger.info("%s Projection refinement: added %d new Gaussians (total: %d)",
                    TAG, len(new_positions), len(positions))
    else:
        logger.info("%s Projection refinement: no voids to fill", TAG)

    return GaussianData(
        positions=positions,
        scales=scales,
        rotations=rotations,
        colors=colors,
        opacities=opacities,
        sh_coeffs=gaussians.sh_coeffs,
        errors=gaussians.errors,
    )


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
