"""Panoramic image to one continuous spherical Gaussian field.

Perspective crops are used only for monocular depth inference. Their depth
predictions are aligned in overlapping regions and blended back into a single
equirectangular depth map before any Gaussians are emitted. This avoids the
six-cloud seams and box-shaped geometry produced by directly merging cube
faces.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from reconstruction_stage import (
    GaussianData,
    _estimate_depth_confidence,
    _infer_metric_depth,
    _postprocess_depth,
    enforce_geometry_quality,
    write_geometry_quality_report,
)

logger = logging.getLogger(__name__)
TAG = "[PANO]"


class PanoramaValidationError(ValueError):
    """Raised when an upload cannot reasonably be treated as a panorama."""


@dataclass(frozen=True)
class CubeFace:
    name: str
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    down: tuple[float, float, float]


# Y-up, right-handed world frame. Image-space y still points down.
FACES = (
    CubeFace("front", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    CubeFace("right", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    CubeFace("back", (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    CubeFace("left", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    CubeFace("zenith", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    CubeFace("nadir", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
)


def validate_panorama(width: int, height: int) -> None:
    """Accept full-sphere and cropped or partial panoramic images."""
    if width < 128 or height < 64:
        raise PanoramaValidationError("Panorama must be at least 128 by 64 pixels.")
    ratio = width / height
    if ratio < 1.2:
        raise PanoramaValidationError(
            f"This image is {ratio:.2f}:1. Local Panorama expects a landscape panoramic image; "
            "use Local Image to 3D for a normal or portrait photograph."
        )


validate_equirectangular = validate_panorama


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _normalise_panorama(source: np.ndarray, target_width: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit a wide panorama into a 2:1 spherical canvas without cropping it."""
    source_h, source_w = source.shape[:2]
    source_ratio = source_w / source_h
    target_width = max(512, int(round(target_width / 32)) * 32)
    target_height = target_width // 2

    if 1.9 <= source_ratio <= 2.1:
        canvas = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        observed = np.ones((target_height, target_width), dtype=np.float32)
        kind = "full_equirectangular"
        note = "The upload already matched a full-sphere equirectangular layout."
    elif source_ratio > 2.1:
        fitted_h = max(1, min(target_height, int(round(target_width / source_ratio))))
        fitted = cv2.resize(source, (target_width, fitted_h), interpolation=cv2.INTER_LANCZOS4)
        top = (target_height - fitted_h) // 2
        bottom = target_height - fitted_h - top
        canvas = cv2.copyMakeBorder(fitted, top, bottom, 0, 0, cv2.BORDER_REPLICATE)
        observed = np.full((target_height, target_width), 0.35, dtype=np.float32)
        observed[top:top + fitted_h] = 1.0
        kind = "vertically_cropped_360"
        note = "The wide panorama was treated as a vertically cropped 360 image; pole bands are extended from its edges."
    else:
        fitted_w = max(1, min(target_width, int(round(target_height * source_ratio))))
        fitted = cv2.resize(source, (fitted_w, target_height), interpolation=cv2.INTER_LANCZOS4)
        left = (target_width - fitted_w) // 2
        right = target_width - fitted_w - left
        canvas = cv2.copyMakeBorder(fitted, 0, 0, left, right, cv2.BORDER_REFLECT_101)
        observed = np.full((target_height, target_width), 0.25, dtype=np.float32)
        observed[:, left:left + fitted_w] = 1.0
        kind = "partial_panorama"
        note = "The partial panorama was extended at its side boundaries; full 360 fidelity needs a complete horizontal sweep."

    blur = max(5, (target_height // 48) | 1)
    observed = np.clip(cv2.GaussianBlur(observed, (blur, blur), 0), 0.0, 1.0)
    profile = {
        "source_width": source_w,
        "source_height": source_h,
        "source_ratio": round(source_ratio, 4),
        "projection_profile": kind,
        "normalised_width": target_width,
        "normalised_height": target_height,
        "normalisation_note": note,
    }
    return canvas, observed, profile


def _erp_rays(height: int, width: int) -> np.ndarray:
    """Return Y-up spherical rays using the standard ERP pixel convention."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    uu, vv = np.meshgrid(u, v)
    theta = (0.5 - uu) * (2.0 * math.pi)
    latitude = (0.5 - vv) * math.pi
    cos_lat = np.cos(latitude)
    return np.stack(
        [cos_lat * np.sin(theta), np.sin(latitude), cos_lat * np.cos(theta)],
        axis=-1,
    ).astype(np.float32)


def _cube_rays(face: CubeFace, size: int, fov_deg: float) -> np.ndarray:
    tangent = math.tan(math.radians(fov_deg) / 2.0)
    xs = np.linspace(-tangent, tangent, size, dtype=np.float32)
    ys = np.linspace(-tangent, tangent, size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    forward = np.asarray(face.forward, dtype=np.float32)
    right = np.asarray(face.right, dtype=np.float32)
    down = np.asarray(face.down, dtype=np.float32)
    rays = forward + grid_x[..., None] * right + grid_y[..., None] * down
    return rays / np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)


def _sample_equirectangular(image: np.ndarray, rays: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    map_x = ((0.5 - longitude / (2.0 * math.pi)) * (width - 1)).astype(np.float32)
    map_y = ((0.5 - latitude / math.pi) * (height - 1)).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def build_cube_faces(image: np.ndarray, size: int, fov_deg: float) -> list[tuple[CubeFace, np.ndarray, np.ndarray]]:
    return [
        (face, _sample_equirectangular(image, rays := _cube_rays(face, size, fov_deg)), rays)
        for face in FACES
    ]


def _sample_face_to_erp(
    face: CubeFace,
    field: np.ndarray,
    erp_rays: np.ndarray,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one perspective field onto the ERP grid with an overlap taper."""
    forward = np.asarray(face.forward, dtype=np.float32)
    right = np.asarray(face.right, dtype=np.float32)
    down = np.asarray(face.down, dtype=np.float32)
    z = np.sum(erp_rays * forward, axis=-1)
    safe_z = np.maximum(z, 1e-5)
    tangent = math.tan(math.radians(fov_deg) / 2.0)
    nx = np.sum(erp_rays * right, axis=-1) / safe_z / tangent
    ny = np.sum(erp_rays * down, axis=-1) / safe_z / tangent
    valid = (z > 0.0) & (np.abs(nx) <= 1.0) & (np.abs(ny) <= 1.0)
    size_h, size_w = field.shape[:2]
    map_x = ((nx + 1.0) * 0.5 * (size_w - 1)).astype(np.float32)
    map_y = ((ny + 1.0) * 0.5 * (size_h - 1)).astype(np.float32)
    sampled = cv2.remap(field.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    edge = np.maximum(np.abs(nx), np.abs(ny))
    taper = np.cos(np.clip(edge, 0.0, 1.0) * math.pi / 2.0) ** 2
    weight = taper.astype(np.float32) * valid.astype(np.float32)
    return sampled, weight


def _align_and_blend_depths(
    face_depths: list[np.ndarray],
    face_confidences: list[np.ndarray],
    weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Align face depth scales in overlaps, then form one ERP depth map."""
    count = len(face_depths)
    equations: list[np.ndarray] = []
    targets: list[float] = []
    for i in range(count):
        for j in range(i + 1, count):
            overlap = (weights[i] > 0.12) & (weights[j] > 0.12)
            overlap &= np.isfinite(face_depths[i]) & np.isfinite(face_depths[j])
            overlap &= (face_depths[i] > 1e-5) & (face_depths[j] > 1e-5)
            if int(overlap.sum()) < 256:
                continue
            delta = np.log(face_depths[j][overlap]) - np.log(face_depths[i][overlap])
            lo, hi = np.quantile(delta, [0.1, 0.9])
            delta = delta[(delta >= lo) & (delta <= hi)]
            row = np.zeros(count, dtype=np.float64)
            row[i] = 1.0
            row[j] = -1.0
            equations.append(row)
            targets.append(float(np.median(delta)))

    anchor = np.zeros(count, dtype=np.float64)
    anchor[0] = 1.0
    equations.append(anchor)
    targets.append(0.0)
    log_scales, *_ = np.linalg.lstsq(np.stack(equations), np.asarray(targets), rcond=None)
    scales = np.exp(np.clip(log_scales, -math.log(3.0), math.log(3.0))).astype(np.float32)

    weighted_log = np.zeros_like(face_depths[0], dtype=np.float64)
    weight_sum = np.zeros_like(face_depths[0], dtype=np.float64)
    confidence_sum = np.zeros_like(face_depths[0], dtype=np.float64)
    for depth, confidence, weight, scale in zip(face_depths, face_confidences, weights, scales):
        aligned = np.maximum(depth * scale, 1e-5)
        combined_weight = weight * np.clip(confidence, 0.05, 1.0)
        weighted_log += combined_weight * np.log(aligned)
        weight_sum += combined_weight
        confidence_sum += combined_weight * confidence

    depth = np.exp(weighted_log / np.maximum(weight_sum, 1e-6)).astype(np.float32)
    confidence = np.clip(confidence_sum / np.maximum(weight_sum, 1e-6), 0.0, 1.0).astype(np.float32)
    return depth, confidence, scales.tolist()


def _rotation_matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert rotations to normalized WXYZ quaternions without per-point Python work."""
    m = matrices.astype(np.float64, copy=False)
    w = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]))
    x = 0.5 * np.sqrt(np.maximum(0.0, 1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]))
    y = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2]))
    z = 0.5 * np.sqrt(np.maximum(0.0, 1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]))
    x = np.copysign(x, m[:, 2, 1] - m[:, 1, 2])
    y = np.copysign(y, m[:, 0, 2] - m[:, 2, 0])
    z = np.copysign(z, m[:, 1, 0] - m[:, 0, 1])
    result = np.stack([w, x, y, z], axis=1).astype(np.float32)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-8)
    return result


def _surface_rotations(rays: np.ndarray) -> np.ndarray:
    normal = -rays
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(np.broadcast_to(world_up, normal.shape), normal)
    degenerate = np.linalg.norm(right, axis=1) < 1e-4
    if degenerate.any():
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right[degenerate] = np.cross(fallback, normal[degenerate])
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-8)
    up = np.cross(normal, right)
    matrices = np.stack([right, up, normal], axis=-1)
    return _rotation_matrices_to_wxyz(matrices)


def estimate_aligned_panorama_depth(
    panorama: np.ndarray,
    *,
    face_size: int | None = None,
    fov_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Estimate one seam-aligned ERP depth map from overlapping cube views.

    This function is shared by the fast spherical fallback and the SHARP-360
    backend.  In SHARP-360 it is only a geometric scale reference; SHARP still
    predicts the actual anisotropic Gaussian geometry for every perspective
    face.
    """
    face_size = face_size or _env_int("LOCAL_PANO_FACE_SIZE", 512, 256, 768)
    fov_deg = fov_deg or _env_float("LOCAL_PANO_FACE_FOV_DEG", 112.0, 96.0, 125.0)
    erp_h, erp_w = panorama.shape[:2]
    rays = _erp_rays(erp_h, erp_w)
    projected_depths: list[np.ndarray] = []
    projected_confidences: list[np.ndarray] = []
    projected_weights: list[np.ndarray] = []

    for face, rgb, _ in build_cube_faces(panorama, face_size, fov_deg):
        logger.info("%s Estimating depth for %s view", TAG, face.name)
        depth = _postprocess_depth(_infer_metric_depth(Image.fromarray(rgb)))
        if not np.isfinite(depth).any():
            raise RuntimeError(f"Depth estimation produced no valid values for the {face.name} view.")
        confidence = _estimate_depth_confidence(depth)
        erp_depth, weight = _sample_face_to_erp(face, depth, rays, fov_deg)
        erp_confidence, _ = _sample_face_to_erp(face, confidence, rays, fov_deg)
        projected_depths.append(erp_depth)
        projected_confidences.append(erp_confidence)
        projected_weights.append(weight)

    return _align_and_blend_depths(
        projected_depths, projected_confidences, projected_weights
    )


def estimate_aligned_panorama_disparity(panorama: np.ndarray, _device=None) -> np.ndarray:
    """Return scale-invariant ERP disparity for SHARP face alignment."""
    depth, _, _ = estimate_aligned_panorama_depth(panorama)
    valid = np.isfinite(depth) & (depth > 1e-5)
    if int(valid.sum()) < 1024:
        raise RuntimeError("Panorama depth alignment produced too few valid pixels")
    median = float(np.median(depth[valid]))
    depth = depth / max(median, 1e-5) * 5.0
    disparity = (1.0 / np.maximum(depth, 1e-5)).astype(np.float32)
    # The SHARP predictor is the next large model in the panorama pipeline.
    # Release Depth Anything here instead of keeping two multi-gigabyte models
    # resident on 12 GB consumer GPUs.
    from reconstruction_stage import unload

    unload()
    return disparity


def reconstruct_panorama(image_path: str | Path, output_dir: Path) -> GaussianData:
    """Lift a panoramic image into a seam-free spherical Gaussian surface."""
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(image_path).convert("RGB"))
    source_h, source_w = source.shape[:2]
    validate_panorama(source_w, source_h)

    face_size = _env_int("LOCAL_PANO_FACE_SIZE", 512, 256, 768)
    fov_deg = _env_float("LOCAL_PANO_FACE_FOV_DEG", 112.0, 96.0, 125.0)
    target_width = _env_int("LOCAL_PANO_ERP_WIDTH", 1536, 768, 3072)
    stride = _env_int("LOCAL_PANO_STRIDE", 1, 1, 6)
    max_points = _env_int("LOCAL_PANO_MAX_POINTS", 1300000, 80000, 2000000)
    target_median_depth = _env_float("LOCAL_PANO_MEDIAN_DEPTH", 3.0, 1.0, 8.0)

    panorama, observed_mask, profile = _normalise_panorama(source, target_width)
    erp_h, erp_w = panorama.shape[:2]
    # Keep sampling spatially uniform. Random confidence/pole pruning creates
    # visible pinholes in low-texture sky; increasing stride gives a regular,
    # coverage-preserving grid when a very large ERP exceeds the point budget.
    required_stride = max(1, int(math.ceil(math.sqrt((erp_w * erp_h) / max_points))))
    stride = max(stride, required_stride)

    rays = _erp_rays(erp_h, erp_w)
    logger.info(
        "%s %dx%d input (%s) -> %dx%d ERP; estimating %d overlapping depth views",
        TAG, source_w, source_h, profile["projection_profile"], erp_w, erp_h, len(FACES),
    )

    depth, confidence, face_scales = estimate_aligned_panorama_depth(
        panorama, face_size=face_size, fov_deg=fov_deg
    )
    fit_region = np.isfinite(depth) & (depth > 1e-5) & (observed_mask > 0.75)
    if int(fit_region.sum()) < 1024:
        fit_region = np.isfinite(depth) & (depth > 1e-5)
    low, median, high = np.quantile(depth[fit_region], [0.01, 0.5, 0.99])
    depth = np.clip(depth, low, high)
    depth = (depth / max(float(median), 1e-5) * target_median_depth).astype(np.float32)

    log_depth = np.log(np.maximum(depth, 1e-5))
    grad_x = np.abs(np.roll(log_depth, -1, axis=1) - np.roll(log_depth, 1, axis=1)) * 0.5
    grad_y = np.abs(np.gradient(log_depth, axis=0))
    edge_confidence = np.exp(-2.5 * np.sqrt(grad_x * grad_x + grad_y * grad_y)).astype(np.float32)
    confidence = np.clip(confidence * edge_confidence, 0.05, 1.0)

    sampled_rays = rays[::stride, ::stride].reshape(-1, 3)
    sampled_depth = depth[::stride, ::stride].reshape(-1)
    sampled_colors = panorama[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
    sampled_conf = confidence[::stride, ::stride].reshape(-1)
    sampled_observed = observed_mask[::stride, ::stride].reshape(-1)
    latitude_cos = np.sqrt(np.clip(1.0 - sampled_rays[:, 1] ** 2, 0.0, 1.0))

    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
    sampled_rays = sampled_rays[valid]
    sampled_depth = sampled_depth[valid]
    sampled_colors = sampled_colors[valid]
    sampled_conf = sampled_conf[valid]
    sampled_observed = sampled_observed[valid]
    latitude_cos = latitude_cos[valid]

    positions = (sampled_rays * sampled_depth[:, None]).astype(np.float32)
    dtheta = 2.0 * math.pi * stride / erp_w
    dphi = math.pi * stride / erp_h
    scale_right = sampled_depth * dtheta * np.maximum(latitude_cos, 0.18) * 1.55
    scale_up = sampled_depth * dphi * 1.55
    scale_normal = np.minimum(scale_right, scale_up) * 0.38
    scales = np.stack([scale_right, scale_up, scale_normal], axis=1).astype(np.float32)
    scales = np.clip(scales, 0.0015, 0.05)
    opacities = np.full_like(sampled_depth, 0.98, dtype=np.float32)
    scores = sampled_conf * (0.6 + 0.4 * sampled_observed)

    if len(positions) > max_points:
        chosen = np.argpartition(scores, -max_points)[-max_points:]
        positions = positions[chosen]
        sampled_colors = sampled_colors[chosen]
        scales = scales[chosen]
        opacities = opacities[chosen]
        sampled_rays = sampled_rays[chosen]

    # The browser camera uses image coordinates (X right, Y down, Z forward).
    # The reconstruction above stays Y-up for spherical depth alignment, then
    # performs one explicit export conversion so the browser view is upright.
    positions[:, 1] *= -1.0
    sampled_rays[:, 1] *= -1.0
    result = GaussianData(
        positions=positions,
        scales=np.log(scales).astype(np.float32),
        rotations=_surface_rotations(sampled_rays),
        colors=sampled_colors.astype(np.float32),
        opacities=opacities[:, None].astype(np.float32),
    )
    result, quality = enforce_geometry_quality(result)
    quality.update(
        {
            **profile,
            "reconstruction": "aligned cube-depth to continuous spherical field",
            "face_count": len(FACES),
            "face_fov_deg": fov_deg,
            "face_size": face_size,
            "face_depth_scales": [round(value, 5) for value in face_scales],
            "sample_stride": stride,
            "translation_note": (
                "A single panorama supports full look-around and modest translation. "
                "Large viewpoint changes reveal surfaces that were never photographed."
            ),
        }
    )
    write_geometry_quality_report(output_dir, quality)
    (output_dir / "panorama_layout.json").write_text(
        json.dumps(
            {
                **profile,
                "source_width": source_w,
                "source_height": source_h,
                "normalised_width": erp_w,
                "normalised_height": erp_h,
                "orientation": "aligned internally in Y-up; exported X-right, Y-down, Z-forward browser camera coordinates",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    Image.fromarray(panorama).save(output_dir / "normalised_panorama.png")
    logger.info("%s Produced %d spherical Gaussians in %.1fs", TAG, result.count, time.time() - started)
    return result

