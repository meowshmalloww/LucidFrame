"""Local equirectangular panorama to Gaussian-splat reconstruction.

This path is deliberately separate from Flash3D.  It turns a *real*
equirectangular panorama into a 360-degree splat shell by lifting spherical
rays with locally inferred depth.  It provides full angular coverage, but a
single panorama still cannot measure disoccluded geometry, so translation is
kept intentionally modest in the viewer.
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
    """Raised when an upload is not an equirectangular panorama."""


@dataclass(frozen=True)
class CubeFace:
    name: str
    forward: tuple[float, float, float]
    right: tuple[float, float, float]
    down: tuple[float, float, float]


FACES = (
    CubeFace("front", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    CubeFace("right", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    CubeFace("back", (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    CubeFace("left", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    CubeFace("up", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    CubeFace("down", (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
)


def validate_equirectangular(width: int, height: int) -> None:
    """Require an equirectangular 2:1-like input rather than any wide image."""
    if width < 256 or height < 128:
        raise PanoramaValidationError("Panorama must be at least 256 by 128 pixels.")
    ratio = width / height
    if not 1.90 <= ratio <= 2.10:
        raise PanoramaValidationError(
            "Local Panorama 360 needs an equirectangular image close to a 2:1 ratio. "
            f"This upload is {ratio:.2f}:1. Use Local Image or export a 360 panorama first."
        )


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


def _cube_rays(face: CubeFace, size: int, fov_deg: float) -> np.ndarray:
    """Return one unit world ray for every pixel in a perspective cube face."""
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
    """Resample RGB equirectangular pixels along unit world directions."""
    height, width = image.shape[:2]
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arcsin(np.clip(-rays[..., 1], -1.0, 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * (width - 1)).astype(np.float32)
    map_y = ((latitude / math.pi + 0.5) * (height - 1)).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def build_cube_faces(image: np.ndarray, size: int, fov_deg: float) -> list[tuple[CubeFace, np.ndarray, np.ndarray]]:
    """Create slightly overlapping RGB cube faces and their world-ray grids."""
    return [(face, _sample_equirectangular(image, rays := _cube_rays(face, size, fov_deg)), rays) for face in FACES]


def _center_weight(rays: np.ndarray, face: CubeFace, fov_deg: float) -> np.ndarray:
    """Prefer overlap samples near a face center to reduce cross-face seams."""
    forward = np.asarray(face.forward, dtype=np.float32)
    cosine = np.clip(np.sum(rays * forward, axis=-1), -1.0, 1.0)
    edge_cosine = math.cos(math.radians(fov_deg) / 2.0)
    return np.clip((cosine - edge_cosine) / max(1.0 - edge_cosine, 1e-5), 0.15, 1.0)


def reconstruct_panorama(image_path: str | Path, output_dir: Path) -> GaussianData:
    """Lift one local equirectangular panorama into a 360-degree Gaussian shell."""
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = source.shape[:2]
    validate_equirectangular(width, height)

    face_size = _env_int("LOCAL_PANO_FACE_SIZE", 384, 192, 768)
    fov_deg = _env_float("LOCAL_PANO_FACE_FOV_DEG", 100.0, 92.0, 110.0)
    stride = _env_int("LOCAL_PANO_STRIDE", 2, 1, 6)
    max_points = _env_int("LOCAL_PANO_MAX_POINTS", 240000, 50000, 600000)
    target_median_depth = _env_float("LOCAL_PANO_MEDIAN_DEPTH", 3.0, 1.0, 8.0)

    logger.info("%s Sampling %dx%d panorama into %d overlapping cube faces", TAG, width, height, len(FACES))
    face_data = build_cube_faces(source, face_size, fov_deg)
    inferred: list[tuple[CubeFace, np.ndarray, np.ndarray, np.ndarray]] = []
    depth_samples: list[np.ndarray] = []

    for face, rgb, rays in face_data:
        logger.info("%s Estimating depth for %s cube face", TAG, face.name)
        raw_depth = _infer_metric_depth(Image.fromarray(rgb))
        depth = _postprocess_depth(raw_depth)
        valid = depth[np.isfinite(depth) & (depth > 1e-5)]
        if valid.size == 0:
            raise RuntimeError(f"Depth estimation produced no valid values for the {face.name} face.")
        inferred.append((face, rgb, rays, depth))
        depth_samples.append(valid)

    global_median = float(np.median(np.concatenate(depth_samples)))
    if not math.isfinite(global_median) or global_median <= 1e-5:
        raise RuntimeError("Panorama depth could not be normalized safely.")

    positions_parts: list[np.ndarray] = []
    colors_parts: list[np.ndarray] = []
    scales_parts: list[np.ndarray] = []
    opacity_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    angular_footprint = 2.0 * math.tan(math.radians(fov_deg) / 2.0) * stride / face_size

    for face, rgb, rays, raw_depth in inferred:
        depth = np.clip(raw_depth / global_median * target_median_depth, 0.35, 9.0).astype(np.float32)
        confidence = _estimate_depth_confidence(depth)
        center = _center_weight(rays, face, fov_deg)
        sampled_rays = rays[::stride, ::stride].reshape(-1, 3)
        sampled_depth = depth[::stride, ::stride].reshape(-1)
        sampled_rgb = rgb[::stride, ::stride].reshape(-1, 3).astype(np.float32) / 255.0
        sampled_confidence = confidence[::stride, ::stride].reshape(-1)
        sampled_center = center[::stride, ::stride].reshape(-1)
        valid = np.isfinite(sampled_depth) & (sampled_depth > 0.05)
        sampled_rays = sampled_rays[valid]
        sampled_depth = sampled_depth[valid]
        sampled_rgb = sampled_rgb[valid]
        sampled_confidence = sampled_confidence[valid]
        sampled_center = sampled_center[valid]
        positions_parts.append(sampled_rays * sampled_depth[:, None])
        colors_parts.append(sampled_rgb)
        scale = np.clip(sampled_depth * angular_footprint * 1.1, 0.004, 0.045).astype(np.float32)
        scales_parts.append(np.repeat(scale[:, None], 3, axis=1))
        opacity_parts.append(np.clip(0.72 + 0.24 * sampled_confidence * sampled_center, 0.55, 0.96))
        score_parts.append(sampled_confidence * sampled_center)

    positions = np.concatenate(positions_parts, axis=0).astype(np.float32)
    colors = np.concatenate(colors_parts, axis=0).astype(np.float32)
    scales = np.concatenate(scales_parts, axis=0).astype(np.float32)
    opacities = np.concatenate(opacity_parts, axis=0).astype(np.float32)
    scores = np.concatenate(score_parts, axis=0).astype(np.float32)

    if len(positions) > max_points:
        chosen = np.argpartition(scores, -max_points)[-max_points:]
        positions, colors, scales, opacities = positions[chosen], colors[chosen], scales[chosen], opacities[chosen]

    rotations = np.zeros((len(positions), 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    result = GaussianData(
        positions=positions,
        scales=np.log(np.clip(scales, 1e-5, 0.05)).astype(np.float32),
        rotations=rotations,
        colors=colors,
        opacities=opacities[:, None],
    )
    result, quality = enforce_geometry_quality(result)
    quality.update({
        "input_projection": "equirectangular",
        "cube_faces": len(FACES),
        "face_fov_deg": fov_deg,
        "face_size": face_size,
        "sample_stride": stride,
        "global_median_depth": global_median,
        "translation_note": "Full angular coverage; hidden geometry remains unobserved from one panorama.",
    })
    write_geometry_quality_report(output_dir, quality)
    (output_dir / "panorama_layout.json").write_text(json.dumps(quality, indent=2))
    logger.info("%s Panorama reconstruction completed in %.1fs with %d Gaussians", TAG, time.time() - started, result.count)
    return result
