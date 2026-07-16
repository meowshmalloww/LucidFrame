"""High-detail local panorama reconstruction through SPAG4D SHARP-360.

Unlike a depth-projected spherical shell, this backend asks SHARP to predict
full anisotropic Gaussians for four or six overlapping horizon views plus
measured zenith/nadir caps. LucidFrame's aligned panoramic depth brings every face
into one consistent scale before the Gaussians are merged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from panorama_spherical_stage import (
    _erp_rays,
    _normalise_panorama,
    _surface_rotations,
    estimate_aligned_panorama_disparity,
    validate_panorama,
)
from reconstruction_stage import GaussianData
from sharp_wrapper import gaussian_data_from_ply, is_available

logger = logging.getLogger(__name__)
TAG = "[SHARP360]"
_CONVERSION_LOCK = threading.Lock()


def _spherical_latitude_mask(means, total_vertical_degrees: float):
    """Keep directions inside a latitude band, independent of face azimuth.

    A perspective rectangle clips with ``|Y/Z|``. At a face corner ``Z`` is
    smaller, so that test removes a triangular wedge before the spherical pole
    cap begins. Measuring ``|Y| / radius`` instead makes the horizon/cap handoff
    a constant latitude around the full sphere.
    """
    import torch

    radius = torch.linalg.vector_norm(means, dim=-1).clamp(min=1e-6)
    half_angle = np.deg2rad(total_vertical_degrees / 2.0)
    return (torch.abs(means[:, 1]) / radius) <= float(np.sin(half_angle))


def _profile_target_width(quality_profile: str) -> int:
    """Resolve ERP inference width, allowing an explicit numeric override."""
    # Detail retains more measured panorama pixels before the fixed-resolution
    # SHARP face crops are sampled. Balanced stays at 2K for lower peak memory.
    default = 2560 if quality_profile == "detail" else 2048
    configured = os.getenv("SHARP360_ERP_WIDTH", "auto").strip().lower()
    if configured in {"", "auto"}:
        value = default
    else:
        try:
            value = int(configured)
        except ValueError:
            logger.warning("%s Invalid SHARP360_ERP_WIDTH=%r; using %d", TAG, configured, default)
            value = default
    return max(1024, min(3072, int(round(value / 32)) * 32))


def _profile_side_count(quality_profile: str) -> int:
    """Resolve the overlap count; Detail uses denser angular evidence."""
    default = 6 if quality_profile == "detail" else 4
    configured = os.getenv("SHARP360_SIDE_COUNT", "auto").strip().lower()
    if configured in {"", "auto"}:
        return default
    try:
        value = int(configured)
    except ValueError:
        logger.warning("%s Invalid SHARP360_SIDE_COUNT=%r; using %d", TAG, configured, default)
        return default
    if value not in {4, 6, 8, 10, 12}:
        logger.warning("%s Unsupported SHARP360_SIDE_COUNT=%d; using %d", TAG, value, default)
        return default
    return value


def _seam_overlap_degrees() -> float:
    """Optional extra angular ownership for learned face merges.

    SPAG4D clips each SHARP prediction exactly at its Voronoi boundary.  That is
    the clean default now that LucidFrame has a depth-aligned coverage underlay.
    The bounded override remains useful for unusually sparse custom models.
    """
    configured = os.getenv("SHARP360_SEAM_OVERLAP_DEGREES", "0").strip()
    try:
        value = float(configured)
    except ValueError:
        logger.warning(
            "%s Invalid SHARP360_SEAM_OVERLAP_DEGREES=%r; using exact ownership",
            TAG,
            configured,
        )
        value = 0.0
    return max(0.0, min(16.0, value))


def _build_coverage_guard(
    panorama: np.ndarray,
    disparity: np.ndarray,
    target_radius: float,
) -> GaussianData:
    """Build a sparse, depth-aligned underlay for residual angular pinholes.

    The learned SHARP Gaussians remain the visible reconstruction.  This layer
    sits slightly behind their aligned median radius and contributes only when
    an uncertain/bright patch has too little opacity or a face boundary lacks a
    sample.  Unlike a flat skybox it uses the same spherical depth field that
    aligns the six learned views, so modest translation still produces parallax.
    """
    height, width = panorama.shape[:2]
    try:
        max_points = int(os.getenv("SHARP360_GUARD_MAX_POINTS", "260000"))
    except ValueError:
        max_points = 260000
    max_points = max(50000, min(350000, max_points))
    stride = max(1, int(np.ceil(np.sqrt((height * width) / max_points))))

    depth = 1.0 / np.maximum(disparity.astype(np.float32), 1e-5)
    valid_depth = np.isfinite(depth) & (depth > 1e-5)
    if int(valid_depth.sum()) < 1024:
        raise RuntimeError("SHARP-360 coverage guard received an invalid depth field")
    low, median, high = np.quantile(depth[valid_depth], [0.01, 0.5, 0.99])
    depth = np.clip(depth, low, high)
    depth *= np.float32((target_radius * 1.08) / max(float(median), 1e-5))
    # This is an underlay, not a second foreground reconstruction. Keep every
    # sample behind the comfortable translation zone so a local depth outlier
    # cannot expand into a large card in front of the learned scene.
    depth = np.clip(depth, target_radius * 0.90, target_radius * 1.65)

    rays = _erp_rays(height, width)[::stride, ::stride].reshape(-1, 3)
    sampled_depth = depth[::stride, ::stride].reshape(-1)
    colors = panorama[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
    finite = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
    rays = rays[finite]
    sampled_depth = sampled_depth[finite]
    colors = colors[finite]

    positions = (rays * sampled_depth[:, None]).astype(np.float32)
    latitude_cos = np.sqrt(np.clip(1.0 - rays[:, 1] ** 2, 0.0, 1.0))
    dtheta = 2.0 * np.pi * stride / width
    dphi = np.pi * stride / height
    # A denser underlay needs less per-splat dilation. This keeps residual
    # coverage while avoiding the soft, oversized cards that become visible
    # during translation.
    scale_right = sampled_depth * dtheta * np.maximum(latitude_cos, 0.18) * 1.38
    scale_up = sampled_depth * dphi * 1.38
    scale_normal = np.minimum(scale_right, scale_up) * 0.25
    linear_scales = np.stack([scale_right, scale_up, scale_normal], axis=1)
    linear_scales = np.clip(linear_scales, 0.002, 0.12).astype(np.float32)

    # Export in the browser's X-right/Y-down/Z-forward camera convention.
    positions[:, 1] *= -1.0
    rays[:, 1] *= -1.0
    opacity = float(os.getenv("SHARP360_GUARD_OPACITY", "0.88"))
    opacity = max(0.5, min(0.99, opacity))
    return GaussianData(
        positions=positions,
        scales=np.log(linear_scales),
        rotations=_surface_rotations(rays),
        colors=colors,
        opacities=np.full((len(positions), 1), opacity, dtype=np.float32),
    )


def _repair_directional_color_outliers(
    gaussians: GaussianData,
    panorama: np.ndarray,
    *,
    chunk_size: int = 500_000,
) -> int:
    """Repair only severe dark predictions that contradict the source direction.

    SHARP occasionally emits near-black DC colors in clipped highlights.  Those
    appear as holes even when geometry exists.  Directional comparison keeps
    ordinary learned shading and dark objects intact; replacement is limited to
    Gaussians whose reference panorama direction is bright and very different.
    """
    height, width = panorama.shape[:2]
    repaired = 0
    colors = gaussians.colors
    for start in range(0, gaussians.count, chunk_size):
        stop = min(start + chunk_size, gaussians.count)
        position = gaussians.positions[start:stop]
        radius = np.maximum(np.linalg.norm(position, axis=1), 1e-6)
        longitude = np.arctan2(position[:, 0], position[:, 2])
        latitude = np.arcsin(np.clip(-position[:, 1] / radius, -1.0, 1.0))
        sample_x = np.mod(
            np.rint((0.5 - longitude / (2.0 * np.pi)) * width).astype(np.int64),
            width,
        )
        sample_y = np.clip(
            np.rint((0.5 - latitude / np.pi) * height).astype(np.int64),
            0,
            height - 1,
        )
        reference = panorama[sample_y, sample_x].astype(np.float32) / 255.0
        predicted = colors[start:stop]
        reference_luma = reference @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        predicted_luma = predicted @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        difference = np.linalg.norm(reference - predicted, axis=1)
        mask = (reference_luma > 0.72) & (predicted_luma < 0.20) & (difference > 0.58)
        if mask.any():
            predicted[mask] = reference[mask] * 0.88 + predicted[mask] * 0.12
            repaired += int(mask.sum())
    return repaired


def _merge_gaussians(primary: GaussianData, underlay: GaussianData) -> GaussianData:
    return GaussianData(
        positions=np.concatenate([primary.positions, underlay.positions], axis=0),
        scales=np.concatenate([primary.scales, underlay.scales], axis=0),
        rotations=np.concatenate([primary.rotations, underlay.rotations], axis=0),
        colors=np.concatenate([primary.colors, underlay.colors], axis=0),
        opacities=np.concatenate([primary.opacities, underlay.opacities], axis=0),
    )


def reconstruct_sharp360(
    image_path: str | Path,
    output_dir: Path,
    quality_profile: str = "balanced",
) -> GaussianData:
    """Convert a wide panorama into aligned, capped SHARP Gaussians."""
    if not is_available():
        raise RuntimeError("SHARP-360 is unavailable because SHARP or CUDA is missing")

    import torch
    import spag4d.sharp360 as spag_sharp360

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = np.asarray(Image.open(image_path).convert("RGB"))
    source_h, source_w = source.shape[:2]
    validate_panorama(source_w, source_h)
    quality_profile = quality_profile if quality_profile in {"balanced", "detail"} else "balanced"
    target_width = _profile_target_width(quality_profile)
    panorama, observed_mask, profile = _normalise_panorama(source, target_width)
    normalized_path = output_dir / "sharp360_normalised_panorama.png"
    Image.fromarray(panorama).save(normalized_path)

    side_count = _profile_side_count(quality_profile)
    seam_overlap = _seam_overlap_degrees()
    requested_caps = os.getenv("SHARP360_INCLUDE_CAPS", "auto").lower()
    source_has_poles = profile["projection_profile"] == "full_equirectangular"
    include_caps = source_has_poles if requested_caps == "auto" else requested_caps == "on"
    ply_path = output_dir / "sharp360.ply"

    def progress(stage: str, current: int, total: int) -> None:
        logger.info("%s %s %d/%d", TAG, stage, current, total)

    # SPAG4D normally uses DA360 for scale alignment.  LucidFrame already has a
    # six-view overlap-aligned metric depth stack installed, so patch only this
    # call site rather than forcing a second 1.5 GB depth model into VRAM.
    with _CONVERSION_LOCK:
        original_depth = spag_sharp360.predict_da360_disparity
        original_border_filter = spag_sharp360.filter_gaussians_by_view_border
        original_cone_filter = spag_sharp360.filter_gaussians_by_cone

        def spherical_border_filter(gaussians, horizontal_degrees, vertical_degrees=None):
            # Horizontal ownership remains a Voronoi strip. Vertical ownership
            # must be measured in spherical latitude: the original perspective
            # |Y/Z| rectangle recedes toward every horizontal face corner and
            # leaves triangular holes before the circular pole cap begins.
            expanded_horizontal = min(178.0, horizontal_degrees + seam_overlap)
            expanded_vertical = (
                None
                if vertical_degrees is None
                else min(178.0, vertical_degrees + 2.0 * seam_overlap)
            )
            filtered = original_border_filter(
                gaussians,
                expanded_horizontal,
                vertical_degrees=None,
            )
            if expanded_vertical is None:
                return filtered

            mask = _spherical_latitude_mask(filtered.mean_vectors[0], expanded_vertical)
            gaussian_type = type(filtered)
            return gaussian_type(
                mean_vectors=filtered.mean_vectors[:, mask],
                singular_values=filtered.singular_values[:, mask],
                quaternions=filtered.quaternions[:, mask],
                colors=filtered.colors[:, mask],
                opacities=filtered.opacities[:, mask],
            )

        def overlapping_cone_filter(gaussians, half_angle_degrees):
            # Do not claim directions outside the actual cap image FOV.
            cap_half_fov = 65.0  # cap_fov_degrees / 2 below
            expanded_half_angle = min(
                cap_half_fov,
                half_angle_degrees + seam_overlap / 2.0,
            )
            return original_cone_filter(gaussians, expanded_half_angle)

        aligned_disparity: np.ndarray | None = None

        def capture_aligned_disparity(panorama_image, device=None):
            nonlocal aligned_disparity
            aligned_disparity = estimate_aligned_panorama_disparity(panorama_image, device)
            return aligned_disparity

        spag_sharp360.predict_da360_disparity = capture_aligned_disparity
        spag_sharp360.filter_gaussians_by_view_border = spherical_border_filter
        spag_sharp360.filter_gaussians_by_cone = overlapping_cone_filter
        try:
            stats = spag_sharp360.convert_sharp360(
                input_path=str(normalized_path),
                output_path=str(ply_path),
                device=torch.device("cuda"),
                side_count=side_count,
                overlap_degrees=12.0,
                include_caps=include_caps,
                cap_fov_degrees=130.0,
                seam_latitude_degrees=29.0,
                progress_callback=progress,
            )
        finally:
            spag_sharp360.predict_da360_disparity = original_depth
            spag_sharp360.filter_gaussians_by_view_border = original_border_filter
            spag_sharp360.filter_gaussians_by_cone = original_cone_filter

    result = gaussian_data_from_ply(ply_path, y_up=True)
    radii = np.linalg.norm(result.positions, axis=1)
    radius_low, radius_high = np.quantile(radii[np.isfinite(radii)], [0.002, 0.998])
    min_opacity = float(os.getenv("SHARP360_MIN_OPACITY", "0.02"))
    keep = (
        (result.opacities[:, 0] >= min_opacity)
        & (radii >= radius_low)
        & (radii <= radius_high)
    )
    result = GaussianData(
        positions=result.positions[keep],
        scales=result.scales[keep],
        rotations=result.rotations[keep],
        colors=result.colors[keep],
        opacities=result.opacities[keep],
    )
    if result.count < 1000:
        raise RuntimeError("SHARP-360 produced too few valid Gaussians")

    # SPAG4D restores the very large pre-alignment radius of wide-angle SHARP
    # faces.  Normalize the merged field back to a room-scale radius so a
    # 30–50 cm camera translation produces visible parallax and scale clamping
    # does not punch holes between distant splats.
    median_radius_before = float(np.median(np.linalg.norm(result.positions, axis=1)))
    target_radius = float(os.getenv("SHARP360_TARGET_RADIUS", "4.0"))
    scale_factor = target_radius / max(median_radius_before, 1e-6)
    result.positions *= np.float32(scale_factor)
    result.scales += np.float32(np.log(scale_factor))

    repaired_color_count = _repair_directional_color_outliers(result, panorama)

    guard_enabled = os.getenv("SHARP360_COVERAGE_GUARD", "on").strip().lower() != "off"
    guard_count = 0
    if guard_enabled and aligned_disparity is not None:
        guard = _build_coverage_guard(panorama, aligned_disparity, target_radius)
        guard_count = guard.count
        result = _merge_gaussians(result, guard)

    report = {
        **profile,
        "backend": "SPAG4D SHARP-360 with LucidFrame aligned depth",
        "quality_profile": quality_profile,
        "license": "SPAG4D MIT; Apple SHARP model is non-commercial research",
        "gaussian_count": result.count,
        "learned_gaussian_count": result.count - guard_count,
        "coverage_guard_count": guard_count,
        "directional_color_repairs": repaired_color_count,
        "predicted_faces": int(stats.get("num_faces", side_count + (2 if include_caps else 0))),
        "horizon_faces": side_count,
        "pole_caps": include_caps,
        "seam_overlap_degrees": seam_overlap,
        "observed_fraction": round(float(np.mean(observed_mask > 0.75)), 5),
        "median_radius_before_normalization": round(median_radius_before, 5),
        "target_median_radius": round(target_radius, 5),
        "metric_scale_factor": round(scale_factor, 8),
        "elapsed_sec": round(time.time() - started, 3),
        "geometry_note": (
            "Each face uses learned anisotropic Gaussians. Unseen surfaces behind foreground "
            "objects are still not measured by a single panorama."
        ),
    }
    (output_dir / "sharp360_quality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info("%s Produced %d Gaussians in %.1fs", TAG, result.count, time.time() - started)
    return result
