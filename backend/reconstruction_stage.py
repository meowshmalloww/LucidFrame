"""
Stage 3: 3D Gaussian Reconstruction.

Primary: Flash3D feed-forward layered Gaussian prediction.
Optional: Multi-view depth fusion for centred object assets only.
Fallback: Single-image depth backprojection.

Outputs raw Gaussian parameters that splat_compiler.py converts to .splat binary.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
TAG = "[RECON]"

# A monocular image has only one measured view. Flash3D produces a compact
# learned continuation for nearby exploration, while Zero123++ remains opt-in
# for centred objects only (see its model card).
RECONSTRUCTION_MODEL = os.getenv("RECONSTRUCTION_MODEL", "sharp").lower()


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(value, minimum) if minimum is not None else value


@dataclass(frozen=True)
class GeometryQualityConfig:
    fov_deg: float
    min_scale: float
    max_scale: float
    max_aspect_ratio: float
    min_opacity: float
    prune_opacity: float

    @classmethod
    def from_env(cls) -> "GeometryQualityConfig":
        min_scale = _env_float("GAUSSIAN_MIN_SCALE", 0.003, 1e-5)
        max_scale = max(_env_float("GAUSSIAN_MAX_SCALE", 0.35, min_scale), min_scale)
        return cls(
            fov_deg=_env_float("RECONSTRUCTION_FOV_DEG", 60.0, 1.0),
            min_scale=min_scale,
            max_scale=max_scale,
            max_aspect_ratio=_env_float("GAUSSIAN_MAX_ASPECT_RATIO", 4.0, 1.0),
            min_opacity=_env_float("GAUSSIAN_MIN_OPACITY", 0.05, 0.0),
            prune_opacity=_env_float("GAUSSIAN_PRUNE_OPACITY", 0.01, 0.0),
        )


GEOMETRY_CONFIG = GeometryQualityConfig.from_env()


@dataclass
class GaussianData:
    """Container for raw Gaussian parameters."""
    positions: np.ndarray      # (N, 3) float32
    scales: np.ndarray         # (N, 3) float32 (log-scale)
    rotations: np.ndarray      # (N, 4) float32 (quaternion wxyz)
    colors: np.ndarray         # (N, 3) float32 (RGB 0-1)
    opacities: np.ndarray      # (N, 1) float32 (0-1)
    sh_coeffs: np.ndarray | None = None  # (N, K, 3) or None
    errors: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.positions)


# ── Geometry Quality ────────────────────────────────────────────────────────

def _geometry_metrics(gaussians: GaussianData) -> dict[str, float | int]:
    if gaussians.count == 0:
        return {"count": 0, "finite_count": 0, "max_aspect_ratio": 0.0}
    scales = np.exp(np.clip(gaussians.scales.astype(np.float32), -20.0, 20.0))
    finite = (
        np.isfinite(gaussians.positions).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(gaussians.rotations).all(axis=1)
        & np.isfinite(gaussians.colors).all(axis=1)
        & np.isfinite(gaussians.opacities.reshape(-1))
    )
    safe_min = np.maximum(scales.min(axis=1), 1e-8)
    aspect_ratios = scales.max(axis=1) / safe_min
    return {
        "count": int(gaussians.count),
        "finite_count": int(finite.sum()),
        "max_aspect_ratio": float(aspect_ratios.max()),
        "median_aspect_ratio": float(np.median(aspect_ratios)),
        "min_linear_scale": float(scales.min()),
        "max_linear_scale": float(scales.max()),
        "min_opacity": float(gaussians.opacities.min()),
        "max_opacity": float(gaussians.opacities.max()),
    }


def enforce_geometry_quality(
    gaussians: GaussianData,
    config: GeometryQualityConfig = GEOMETRY_CONFIG,
) -> tuple[GaussianData, dict[str, float | int]]:
    before = _geometry_metrics(gaussians)
    if gaussians.count == 0:
        return gaussians, {"before": before, "after": before, "pruned_count": 0}

    opacities = gaussians.opacities.astype(np.float32).reshape(-1, 1)
    scales = np.exp(np.clip(gaussians.scales.astype(np.float32), -20.0, 20.0))
    finite = (
        np.isfinite(gaussians.positions).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(gaussians.rotations).all(axis=1)
        & np.isfinite(gaussians.colors).all(axis=1)
        & np.isfinite(opacities[:, 0])
    )
    keep = finite & (opacities[:, 0] >= config.prune_opacity)
    if not keep.any():
        empty = GaussianData(
            positions=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            rotations=np.empty((0, 4), dtype=np.float32),
            colors=np.empty((0, 3), dtype=np.float32),
            opacities=np.empty((0, 1), dtype=np.float32),
            errors=gaussians.errors + ["Geometry quality gate removed all Gaussians"],
        )
        return empty, {"before": before, "after": _geometry_metrics(empty), "pruned_count": int(gaussians.count)}

    positions = gaussians.positions[keep].astype(np.float32)
    rotations = gaussians.rotations[keep].astype(np.float32)
    colors = np.clip(gaussians.colors[keep].astype(np.float32), 0.0, 1.0)
    opacities = np.clip(opacities[keep], config.min_opacity, 1.0)
    scales = np.clip(scales[keep], config.min_scale, config.max_scale)
    largest = scales.max(axis=1, keepdims=True)
    scales = np.maximum(scales, largest / config.max_aspect_ratio)
    scales = np.clip(scales, config.min_scale, config.max_scale)
    rotation_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    invalid_rotations = rotation_norms[:, 0] < 1e-8
    rotations = rotations / np.where(rotation_norms > 1e-8, rotation_norms, 1.0)
    rotations[invalid_rotations] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    sanitized = GaussianData(
        positions=positions,
        scales=np.log(scales).astype(np.float32),
        rotations=rotations.astype(np.float32),
        colors=colors,
        opacities=opacities.astype(np.float32),
        sh_coeffs=gaussians.sh_coeffs[keep] if gaussians.sh_coeffs is not None else None,
        errors=gaussians.errors,
    )
    report = {
        "before": before,
        "after": _geometry_metrics(sanitized),
        "pruned_count": int(gaussians.count - sanitized.count),
    }
    return sanitized, report


def write_geometry_quality_report(output_dir: Path, report: dict[str, object]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry_quality.json").write_text(json.dumps(report, indent=2))


# ── Hardware Checks ─────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Depth Post-Processing ───────────────────────────────────────────────────

def _bilateral_smooth_depth(depth: np.ndarray) -> np.ndarray:
    """Edge-aware bilateral smoothing on normalised depth, then rescale."""
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return depth.astype(np.float32)
    normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    smoothed = cv2.bilateralFilter(normalized, d=5, sigmaColor=30, sigmaSpace=30)
    return (smoothed.astype(np.float32) / 255.0 * (d_max - d_min) + d_min).astype(np.float32)


def _median_filter_depth(depth: np.ndarray) -> np.ndarray:
    """5x5 median filter to remove depth spikes."""
    return cv2.medianBlur(depth.astype(np.float32), 5)


def _fill_depth_holes(depth: np.ndarray) -> tuple[np.ndarray, int]:
    """Inpaint pixels that are zero or extreme outliers."""
    valid_mask = (depth > 1e-4) & np.isfinite(depth)
    hole_mask = (~valid_mask).astype(np.uint8) * 255
    n_holes = int(hole_mask.sum() // 255)
    if n_holes == 0:
        return depth, 0
    result = cv2.inpaint(depth.astype(np.float32), hole_mask, 5, cv2.INPAINT_TELEA)
    return result, n_holes


def _fix_normal_inconsistencies(depth: np.ndarray) -> np.ndarray:
    """Smooth pixels where surface normal differs sharply from neighbours without a depth edge."""
    zy, zx = np.gradient(depth.astype(np.float64))
    mag = np.sqrt(zx ** 2 + zy ** 2)
    mag_threshold = float(np.percentile(mag, 95))
    high_grad = mag > mag_threshold
    result = depth.copy()
    if high_grad.any():
        result[high_grad] = _median_filter_depth(result)[high_grad]
    return result.astype(np.float32)


def _postprocess_depth(depth: np.ndarray) -> np.ndarray:
    """Full post-processing pipeline: holes → light bilateral → normal consistency."""
    depth, n_holes = _fill_depth_holes(depth)
    logger.info("%s Depth holes filled: %d", TAG, n_holes)
    depth = _bilateral_smooth_depth(depth)
    depth = _fix_normal_inconsistencies(depth)
    return depth


# ── Depth Estimation ────────────────────────────────────────────────────────

_DEPTH_MODEL: Any = None
_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def _depth_model_source() -> tuple[str, bool]:
    """Resolve the depth model without requiring a network round trip.

    Hugging Face caches each revision as a self-contained snapshot. Resolving
    that snapshot first keeps an already-downloaded model usable when a laptop
    is offline and avoids a lookup for optional processor metadata. Set
    DEPTH_MODEL_ALLOW_DOWNLOAD=on only when an intentional download is allowed.
    """
    allow_download = os.getenv("DEPTH_MODEL_ALLOW_DOWNLOAD", "off").lower() == "on"
    try:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=_DEPTH_MODEL_ID,
            local_files_only=not allow_download,
        )
        return snapshot, True
    except Exception as exc:
        if allow_download:
            raise RuntimeError(
                f"Could not resolve {_DEPTH_MODEL_ID} after an allowed download attempt: {exc}"
            ) from exc
        raise RuntimeError(
            "Depth model is not available in the local Hugging Face cache. "
            "Connect once and set DEPTH_MODEL_ALLOW_DOWNLOAD=on to fetch it, "
            "then keep future runs offline."
        ) from exc


def _load_depth_model():
    """Load Depth Anything V2 Metric Indoor model (lazy)."""
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_source, local_files_only = _depth_model_source()
    logger.info("%s Loading Depth Anything V2 Metric Indoor from %s...", TAG, model_source)
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained(
        model_source,
        local_files_only=local_files_only,
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        dtype=torch.float16,
    ).to("cuda").eval()

    _DEPTH_MODEL = (processor, model)
    logger.info("%s Depth Anything V2 Metric Indoor loaded in %.1fs", TAG, time.time() - t0)
    return _DEPTH_MODEL


def _infer_metric_depth(pil_img) -> np.ndarray:
    """Run the metric depth model and return a depth map in meters (H×W float32)."""
    processor, model = _load_depth_model()
    import torch

    inputs = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to("cuda")

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    pred = outputs.predicted_depth
    if pred.dim() == 4:
        pred = pred.squeeze(1)

    w, h = pil_img.size
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)

    depth_meters = pred.squeeze(0).cpu().numpy().astype(np.float32)
    return depth_meters


def _estimate_intrinsics(w: int, h: int, fov_deg: float = 55.0) -> tuple[float, float, float, float]:
    """Estimate pinhole intrinsics from image size and assumed horizontal FOV."""
    fx = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx  # square pixels
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


def _compute_normals_from_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Compute surface normals from a depth map using central differences."""
    h, w = depth.shape
    dz_dy = np.zeros_like(depth)
    dz_dx = np.zeros_like(depth)
    dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / 2.0
    dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / 2.0

    z = depth + 1e-6
    gx = dz_dx * fx / z
    gy = dz_dy * fy / z

    nx, ny, nz = -gx, -gy, np.ones_like(depth)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return normals


def _normal_to_quaternion(normal: np.ndarray) -> np.ndarray:
    """Convert a surface normal (pointing toward camera) to a quaternion (wxyz)."""
    forward = np.array([0.0, 0.0, 1.0])
    n = normal / (np.linalg.norm(normal) + 1e-8)
    dot = np.clip(np.dot(forward, n), -1.0, 1.0)
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    angle = math.acos(dot)
    axis = np.cross(forward, n)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    s = math.sin(angle / 2.0)
    return np.array([
        math.cos(angle / 2.0),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
    ], dtype=np.float32)


def _estimate_depth_confidence(depth: np.ndarray) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(depth.astype(np.float32))
    gradient = np.hypot(dz_dx, dz_dy)
    laplacian = np.abs(cv2.Laplacian(depth.astype(np.float32), cv2.CV_32F))
    gradient_scale = float(np.percentile(gradient, 90)) + 1e-6
    laplacian_scale = float(np.percentile(laplacian, 90)) + 1e-6
    confidence = np.exp(-gradient / gradient_scale - 0.5 * laplacian / laplacian_scale)
    return np.clip(confidence, 0.05, 1.0).astype(np.float32)


# ── Single-Image Depth Backprojection (fallback) ────────────────────────────

def reconstruct_single_depth(
    image_path: str,
    output_dir: Path,
) -> GaussianData:
    """
    Single-image depth backprojection — the most robust fallback.
    Produces a depth-relief point cloud from one image.
    """
    t0 = time.time()
    errors: list[str] = []

    if not _has_cuda():
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]),
            errors=["CUDA not available"],
        )

    import torch
    from PIL import Image

    try:
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size

        logger.info("%s Estimating metric depth for %dx%d image...", TAG, w, h)
        depth_raw = _infer_metric_depth(pil_img)
        depth_map = _postprocess_depth(depth_raw)
        depth_safe = np.clip(depth_map, 0.1, 20.0).astype(np.float32)

        fx, fy, cx, cy = _estimate_intrinsics(w, h, fov_deg=GEOMETRY_CONFIG.fov_deg)
        img_np = np.array(pil_img).astype(np.float32) / 255.0

        total_pixels = w * h
        step = 2 if total_pixels > 500_000 else 1
        logger.info("%s Using step=%d (~%d Gaussians)", TAG, step, total_pixels // (step * step))

        ys, xs = np.mgrid[0:h:step, 0:w:step]
        z = depth_safe[ys, xs]
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy

        positions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1).astype(np.float32)
        colors = img_np[ys, xs].reshape(-1, 3).astype(np.float32)
        sampled_depths = z.flatten().astype(np.float32)
        confidence = _estimate_depth_confidence(depth_map)
        sampled_confidence = confidence[ys, xs].reshape(-1)

        z_vals = positions[:, 2]
        z_thresh = np.percentile(z_vals, 99.0)
        valid = (z_vals < z_thresh) & (sampled_confidence >= 0.05)
        positions = positions[valid]
        colors = colors[valid]
        sampled_depths = sampled_depths[valid]
        sampled_confidence = sampled_confidence[valid]

        # Center in X/Y, scale to ~5 units deep
        center_xy = np.array([
            float(np.median(positions[:, 0])),
            float(np.median(positions[:, 1])),
            0.0,
        ], dtype=np.float32)
        positions -= center_xy
        z_extent = positions[:, 2].max() - positions[:, 2].min()
        scale_factor = 5.0 / z_extent if z_extent > 0.1 else 1.0
        positions = positions * scale_factor

        n = len(positions)
        logger.info("%s Single-depth: %d Gaussians, scale=%.2f", TAG, n, scale_factor)

        # Normals and rotations
        normals = _compute_normals_from_depth(depth_map, fx, fy)
        sampled_normals = normals[ys, xs].reshape(-1, 3)[valid]
        rotations = np.stack([_normal_to_quaternion(nm) for nm in sampled_normals], axis=0).astype(np.float32)

        # Adaptive scales
        pixel_footprint = sampled_depths * step * scale_factor / fx
        adaptive = 1.0 + 0.5 * (1.0 - sampled_confidence)
        gaussian_xy = np.clip(pixel_footprint * 1.3 * adaptive, 0.003, 0.20).astype(np.float32)
        gaussian_z = np.clip(gaussian_xy * 0.6, 0.004, 0.12).astype(np.float32)
        linear_scales = np.column_stack([gaussian_xy, gaussian_xy, gaussian_z])
        opacities = np.clip(0.85 + 0.13 * sampled_confidence, 0.85, 0.98).astype(np.float32)

        # Jitter to break grid
        rng = np.random.default_rng(seed=42)
        positions += rng.normal(0, 1, size=positions.shape).astype(np.float32) * (gaussian_xy[:, None] * 0.15)

        scales_log = np.log(np.clip(linear_scales, 1e-6, 0.05)).astype(np.float32)

        result = GaussianData(
            positions=positions,
            scales=scales_log,
            rotations=rotations,
            colors=colors,
            opacities=opacities[:, None],
            errors=errors,
        )
        result, quality_report = enforce_geometry_quality(result)
        quality_report["reconstruction_fov_deg"] = GEOMETRY_CONFIG.fov_deg
        write_geometry_quality_report(output_dir, quality_report)
        logger.info("%s Single-depth done in %.1fs (%d Gaussians)", TAG, time.time() - t0, result.count)
        return result

    except Exception as exc:
        import traceback
        logger.error("%s Single-depth failed: %s\n%s", TAG, exc, traceback.format_exc())
        errors.append(str(exc))
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]),
            errors=errors,
        )


# ── Multi-View Depth Fusion ─────────────────────────────────────────────────

# Zero123++ v1.2 camera poses: 6 views with known azimuth/elevation
# Azimuth is relative to input view, elevation is absolute
# FOV is 30° for generated views
_ZERO123_V1_2_POSES = [
    {"azimuth": 30,  "elevation": 20},
    {"azimuth": 90,  "elevation": -10},
    {"azimuth": 150, "elevation": 20},
    {"azimuth": 210, "elevation": -10},
    {"azimuth": 270, "elevation": 20},
    {"azimuth": 330, "elevation": -10},
]
_ZERO123_FOV_DEG = 30.0

# Normalized depth range — all views scaled to this range for consistency
_DEPTH_NEAR = 0.5
_DEPTH_FAR = 2.5
_CAM_DISTANCE = (_DEPTH_NEAR + _DEPTH_FAR) / 2.0  # 1.5


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to a consistent range using percentile-based normalization.

    This fixes cross-view depth scale inconsistency — metric depth models
    give different absolute scales for different viewpoints. By normalizing
    each view to the same [near, far] range, all point clouds align.
    """
    d_min = float(np.percentile(depth, 5))
    d_max = float(np.percentile(depth, 95))
    if d_max - d_min < 1e-6:
        return np.full_like(depth, _CAM_DISTANCE, dtype=np.float32)
    normalized = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    return (normalized * (_DEPTH_FAR - _DEPTH_NEAR) + _DEPTH_NEAR).astype(np.float32)


def _voxel_downsample(
    positions: np.ndarray, colors: np.ndarray, voxel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel downsample a point cloud — average points within each voxel.

    This preserves spatial coverage much better than random subsampling.
    """
    voxel_idx = np.floor(positions / voxel_size).astype(np.int64)
    # Hash voxel indices to 1D keys
    keys = voxel_idx[:, 0] * 10000000 + voxel_idx[:, 1] * 100000 + voxel_idx[:, 2]
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    n_voxels = len(unique_keys)

    summed_pos = np.zeros((n_voxels, 3), dtype=np.float64)
    summed_col = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.int64)
    np.add.at(summed_pos, inverse, positions)
    np.add.at(summed_col, inverse, colors)
    np.add.at(counts, inverse, 1)

    avg_pos = (summed_pos / counts[:, None]).astype(np.float32)
    avg_col = (summed_col / counts[:, None]).astype(np.float32)
    return avg_pos, avg_col


def _azimuth_elevation_to_extrinsics(azimuth_deg: float, elevation_deg: float, distance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera extrinsics (R, T) for a view at given azimuth/elevation.
    Camera looks at the origin (scene center). World coordinate system:
    +X right, +Y down, +Z forward (into the scene).

    Returns (R_3x3, T_3) such that world_to_cam = [R | T].
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    cx = distance * math.cos(el) * math.sin(az)
    cy = distance * math.sin(el)
    cz = distance * math.cos(el) * math.cos(az)

    forward = np.array([-cx, -cy, -cz], dtype=np.float32)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0, -1, 0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)

    # CV convention: camera X=right, Y=down, Z=forward (into scene)
    R = np.stack([right, -up, forward], axis=0).astype(np.float32)
    T = -R @ np.array([cx, cy, cz], dtype=np.float32)
    return R, T


def reconstruct_multiview_depth(
    view_paths: list[str],
    output_dir: Path,
) -> GaussianData:
    """
    Multi-view depth fusion reconstruction.

    All views (input + 6 Zero123++ generated) are treated uniformly:
    - Each camera orbits the origin at _CAM_DISTANCE
    - Input view: azimuth=0, elevation=0 (front view)
    - Generated views: known azimuth/elevation from Zero123++ v1.2
    - Depth is normalized per-view to [0.5, 2.5] for cross-view consistency
    - Voxel downsampling preserves spatial coverage
    """
    t0 = time.time()
    errors: list[str] = []

    if not _has_cuda():
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]),
            errors=["CUDA not available"],
        )

    from PIL import Image

    try:
        _load_depth_model()

        # All 7 views: input (az=0, el=0) + 6 Zero123++ views
        all_poses = [{"azimuth": 0, "elevation": 0}] + _ZERO123_V1_2_POSES
        n_views = min(len(view_paths), len(all_poses))

        all_positions = []
        all_colors = []

        for vi in range(n_views):
            pose = all_poses[vi]
            try:
                pil_view = Image.open(view_paths[vi]).convert("RGB")
                vw, vh = pil_view.size
                img_view = np.array(pil_view).astype(np.float32) / 255.0

                logger.info("%s [MV] View %d: az=%d el=%d (%dx%d)",
                            TAG, vi, pose["azimuth"], pose["elevation"], vw, vh)

                # Depth estimation + normalization
                depth_raw = _infer_metric_depth(pil_view)
                depth_norm = _normalize_depth(depth_raw)

                # Intrinsics — input view uses config FOV, generated use Zero123++ FOV
                fov = GEOMETRY_CONFIG.fov_deg if vi == 0 else _ZERO123_FOV_DEG
                fx, fy, cx, cy = _estimate_intrinsics(vw, vh, fov_deg=fov)

                # Backproject to camera space
                ys, xs = np.mgrid[0:vh, 0:vw]
                z = depth_norm
                x = (xs - cx) * z / fx
                y = (ys - cy) * z / fy
                pts_cam = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1).astype(np.float32)
                cols = img_view.reshape(-1, 3)

                # Camera extrinsics: world_to_cam = [R | T]
                R, T = _azimuth_elevation_to_extrinsics(
                    pose["azimuth"], pose["elevation"], _CAM_DISTANCE
                )

                # Transform to world: world = R^T @ (cam - T)
                pts_world = (pts_cam - T) @ R.T

                # Remove extreme outliers (beyond 3x camera distance from origin)
                dists = np.linalg.norm(pts_world, axis=1)
                valid = dists < _CAM_DISTANCE * 3.0
                pts_world = pts_world[valid]
                cols = cols[valid]

                logger.info("%s [MV] View %d: %d valid points", TAG, vi, len(pts_world))
                all_positions.append(pts_world)
                all_colors.append(cols)

            except Exception as exc:
                logger.warning("%s [MV] View %d failed: %s", TAG, vi, exc)
                errors.append(f"View {vi}: {exc}")

        if not all_positions:
            raise RuntimeError("All views failed depth estimation")

        # ── Merge all point clouds ────────────────────────────────────────
        positions = np.concatenate(all_positions, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        # Center at origin
        center = np.median(positions, axis=0)
        positions = positions - center

        # Voxel downsample to ~50K points for manageable optimization
        extent = float(np.max(np.linalg.norm(positions, axis=1)))
        voxel_size = max(extent / 80.0, 0.01)
        positions, colors = _voxel_downsample(positions, colors, voxel_size)

        # Scale to fit in ~5 unit sphere
        extent = float(np.max(np.linalg.norm(positions, axis=1)))
        scale_factor = 5.0 / extent if extent > 0.1 else 1.0
        positions = positions * scale_factor

        n = len(positions)
        logger.info("%s [MV] Merged: %d points from %d views, voxel=%.3f, scale=%.2f",
                    TAG, n, len(all_positions), voxel_size, scale_factor)

        # ── Convert to Gaussians ──────────────────────────────────────────
        # Uniform scales based on voxel size
        gaussian_size = voxel_size * scale_factor * 1.5
        gaussian_xy = np.full(n, gaussian_size, dtype=np.float32)
        gaussian_z = np.full(n, gaussian_size * 0.6, dtype=np.float32)
        linear_scales = np.column_stack([gaussian_xy, gaussian_xy, gaussian_z])

        rotations = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        opacities = np.full(n, 0.9, dtype=np.float32)

        # Small jitter to break grid pattern
        rng = np.random.default_rng(seed=42)
        positions += rng.normal(0, 1, size=positions.shape).astype(np.float32) * (gaussian_size * 0.1)

        scales_log = np.log(np.clip(linear_scales, 1e-6, 0.05)).astype(np.float32)

        gen_time = time.time() - t0
        logger.info("%s [MV] Multi-view depth fusion done in %.1fs (%d Gaussians)",
                    TAG, gen_time, n)

        result = GaussianData(
            positions=positions.astype(np.float32),
            scales=scales_log,
            rotations=rotations,
            colors=colors.astype(np.float32),
            opacities=opacities[:, None].astype(np.float32),
            errors=errors,
        )

        result, quality_report = enforce_geometry_quality(result)
        quality_report["reconstruction_fov_deg"] = GEOMETRY_CONFIG.fov_deg
        write_geometry_quality_report(output_dir, quality_report)
        logger.info("%s [MV] Geometry quality: %d kept, %d pruned",
                    TAG, result.count, quality_report["pruned_count"])

        return result

    except Exception as exc:
        import traceback
        logger.error("%s [MV] Multi-view depth fusion failed: %s\n%s", TAG, exc, traceback.format_exc())
        errors.append(str(exc))
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]),
            errors=errors,
        )


# ── Main Entry Point ────────────────────────────────────────────────────────

def reconstruct(
    view_paths: list[str],
    output_dir: Path,
) -> GaussianData:
    """
    Main entry point: reconstruct 3D Gaussians from multi-view images.

    Default: sharp (high-resolution metric Gaussian prediction for nearby views)
    Optional: multiview_depth (Zero123++ views for centred object assets only)
    Optional: flash3d (feed-forward Gaussian prediction)
    """
    model = RECONSTRUCTION_MODEL
    result: GaussianData | None = None

    if model == "multiview_depth":
        if len(view_paths) > 1:
            result = reconstruct_multiview_depth(view_paths, output_dir)
        else:
            logger.info("%s Only 1 view — using single_depth", TAG)
            result = reconstruct_single_depth(view_paths[0], output_dir)
        if (result is None or result.count == 0) and view_paths:
            logger.warning("%s multiview_depth failed, falling back to single_depth", TAG)
            result = reconstruct_single_depth(view_paths[0], output_dir)

        # Run differentiable optimization to refine Gaussians
        if result is not None and result.count > 0 and len(view_paths) > 1:
            opt_iters = int(os.getenv("OPTIMIZATION_ITERATIONS", "500"))
            if opt_iters > 0:
                # Free depth model VRAM before optimization
                global _DEPTH_MODEL
                if _DEPTH_MODEL is not None:
                    del _DEPTH_MODEL
                    _DEPTH_MODEL = None
                    if _has_cuda():
                        import torch
                        torch.cuda.empty_cache()

                logger.info("%s Running differentiable optimization (%d iters)...", TAG, opt_iters)
                result = optimize_gaussians(result, view_paths, output_dir, iterations=opt_iters)

    elif model == "sharp":
        if view_paths:
            try:
                from sharp_wrapper import reconstruct_sharp

                result = reconstruct_sharp(view_paths[0], output_dir)
            except Exception as exc:
                logger.warning("%s SHARP failed: %s — falling back to Flash3D", TAG, exc)
                result = None
        if result is None or result.count == 0:
            if view_paths:
                try:
                    from flash3d_wrapper import reconstruct_flash3d

                    result = reconstruct_flash3d(view_paths[0], output_dir)
                except Exception as exc:
                    logger.warning("%s Flash3D fallback failed: %s", TAG, exc)
                    result = None

    elif model == "flash3d":
        if view_paths:
            try:
                from flash3d_wrapper import reconstruct_flash3d
                result = reconstruct_flash3d(view_paths[0], output_dir)
            except Exception as exc:
                logger.warning("%s Flash3D failed: %s — falling back to single_depth", TAG, exc)
                result = None
        if result is None or result.count == 0:
            logger.warning("%s Flash3D failed, falling back to single_depth", TAG)
            if view_paths:
                result = reconstruct_single_depth(view_paths[0], output_dir)

    elif model == "depth_direct":
        if view_paths:
            result = reconstruct_single_depth(view_paths[0], output_dir)

    else:
        # Unknown model: keep the safe, locally measured default.
        logger.warning("%s Unknown reconstruction model '%s'; using depth_direct", TAG, model)
        if view_paths:
            result = reconstruct_single_depth(view_paths[0], output_dir)

    # Final fallback
    if result is None or result.count == 0:
        if view_paths:
            result = reconstruct_single_depth(view_paths[0], output_dir)
        else:
            return GaussianData(
                positions=np.array([]), scales=np.array([]),
                rotations=np.array([]), colors=np.array([]),
                opacities=np.array([]),
                errors=["No view paths provided"],
            )

    return result


# ── Differentiable 3DGS Optimization ────────────────────────────────────────

def _ssim_loss(pred: "torch.Tensor", target: "torch.Tensor", win_size: int = 11) -> "torch.Tensor":
    """Differentiable SSIM loss (1 - SSIM). Simplified — no Gaussian window."""
    import torch
    import torch.nn.functional as F

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Use average pooling as a simple box filter
    pad = win_size // 2
    mu_p = F.avg_pool2d(pred.permute(2, 0, 1).unsqueeze(0), win_size, 1, pad).squeeze(0).permute(1, 2, 0)
    mu_t = F.avg_pool2d(target.permute(2, 0, 1).unsqueeze(0), win_size, 1, pad).squeeze(0).permute(1, 2, 0)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.avg_pool2d((pred ** 2).permute(2, 0, 1).unsqueeze(0), win_size, 1, pad).squeeze(0).permute(1, 2, 0) - mu_p_sq
    sigma_t_sq = F.avg_pool2d((target ** 2).permute(2, 0, 1).unsqueeze(0), win_size, 1, pad).squeeze(0).permute(1, 2, 0) - mu_t_sq
    sigma_pt = F.avg_pool2d((pred * target).permute(2, 0, 1).unsqueeze(0), win_size, 1, pad).squeeze(0).permute(1, 2, 0) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C1)) / (
        (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    )
    return 1.0 - ssim_map.mean()


def optimize_gaussians(
    gaussians: GaussianData,
    view_paths: list[str],
    output_dir: Path,
    iterations: int = 500,
    render_size: int = 256,
    lr_means: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacities: float = 5e-2,
    lr_colors: float = 2.5e-3,
    lambda_dssim: float = 0.2,
    densify_interval: int = 100,
    densify_until_iter: int = 400,
    prune_opacity: float = 0.005,
    gradient_threshold: float = 0.0002,
) -> GaussianData:
    """
    Differentiable 3DGS optimization loop using our PyTorch renderer.

    This is the key quality step that Rem and other projects use:
    1. Render Gaussians from each view angle using torch_splat
    2. Compare rendered image to the target photo (L1 + SSIM loss)
    3. Backpropagate through the renderer to update Gaussian parameters
    4. Densify: clone high-gradient small Gaussians, split large ones
    5. Prune: remove low-opacity Gaussians

    The camera poses match the Zero123++ view arrangement.
    """
    import torch

    if not _has_cuda() or gaussians.count == 0:
        return gaussians

    device = torch.device("cuda")
    t0 = time.time()
    logger.info("%s [OPT] Starting 3DGS optimization: %d Gaussians, %d iterations",
                TAG, gaussians.count, iterations)

    # Initialize trainable parameters from GaussianData
    # Subsample if too many — PyTorch differentiable renderer can't handle 1.8M Gaussians
    max_opt_gaussians = 10_000
    if gaussians.count > max_opt_gaussians:
        logger.info("%s [OPT] Subsampling %d → %d Gaussians for optimization",
                    TAG, gaussians.count, max_opt_gaussians)
        idx = np.random.default_rng(seed=42).choice(
            gaussians.count, max_opt_gaussians, replace=False
        )
        positions_init = gaussians.positions[idx]
        scales_init = gaussians.scales[idx]
        rotations_init = gaussians.rotations[idx]
        colors_init = gaussians.colors[idx]
    else:
        positions_init = gaussians.positions
        scales_init = gaussians.scales
        rotations_init = gaussians.rotations
        colors_init = gaussians.colors

    n_opt = len(positions_init)
    means = torch.tensor(positions_init, dtype=torch.float32, device=device, requires_grad=True)
    log_scales = torch.tensor(scales_init, dtype=torch.float32, device=device, requires_grad=True)
    quats = torch.tensor(rotations_init, dtype=torch.float32, device=device, requires_grad=True)
    logit_opas = torch.full((n_opt,), 2.0, dtype=torch.float32, device=device, requires_grad=True)
    raw_colors = torch.tensor(colors_init, dtype=torch.float32, device=device, requires_grad=True)

    # ── Load training views with camera poses ───────────────────────────
    # Must match reconstruction: all views orbit origin at _CAM_DISTANCE
    all_poses = [{"azimuth": 0, "elevation": 0}] + _ZERO123_V1_2_POSES
    train_data = []
    for vi, vp in enumerate(view_paths[:7]):
        img = cv2.imread(vp)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img = img_rgb.shape[:2]

        # Resize to render_size for optimization speed
        if max(h_img, w_img) > render_size:
            scale = render_size / max(h_img, w_img)
            img_rgb = cv2.resize(img_rgb, (int(w_img * scale), int(h_img * scale)))

        img_t = torch.tensor(img_rgb, dtype=torch.float32, device=device) / 255.0
        h_r, w_r = img_t.shape[:2]

        # Camera intrinsics — match reconstruction FOV per view
        pose = all_poses[vi]
        fov = GEOMETRY_CONFIG.fov_deg if vi == 0 else _ZERO123_FOV_DEG
        fx = (w_r / 2.0) / math.tan(math.radians(fov) / 2.0)
        K = torch.tensor([
            [fx, 0, w_r / 2],
            [0, fx, h_r / 2],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)

        # Camera extrinsics — same as reconstruction: orbit origin at _CAM_DISTANCE
        R_np, T_np = _azimuth_elevation_to_extrinsics(
            pose["azimuth"], pose["elevation"], _CAM_DISTANCE
        )
        ext = torch.eye(4, dtype=torch.float32, device=device)
        ext[:3, :3] = torch.tensor(R_np, device=device)
        ext[:3, 3] = torch.tensor(T_np, device=device)

        train_data.append({
            "image": img_t,
            "K": K,
            "ext": ext,
            "H": h_r,
            "W": w_r,
        })

    if len(train_data) == 0:
        logger.warning("%s [OPT] No valid training images — skipping", TAG)
        return gaussians

    logger.info("%s [OPT] %d training views loaded", TAG, len(train_data))

    # ── Optimizer ────────────────────────────────────────────────────────
    def make_optimizer():
        return torch.optim.Adam([
            {"params": [means], "lr": lr_means},
            {"params": [log_scales], "lr": lr_scales},
            {"params": [quats], "lr": lr_quats},
            {"params": [logit_opas], "lr": lr_opacities},
            {"params": [raw_colors], "lr": lr_colors},
        ])

    optimizer = make_optimizer()

    # Gradient accumulator for densification
    grad_accum = torch.zeros(n_opt, device=device)
    grad_count = torch.zeros(n_opt, device=device)

    from torch_splat import render_simple
    import gc

    best_loss = float("inf")
    last_loss = float("inf")
    n_gaussians = n_opt

    # Disable GC during optimization — in threaded contexts (asyncio.to_thread),
    # Python's GC can collect autograd graph nodes between forward and backward,
    # causing "element 0 of tensors does not require grad" errors.
    gc_enabled = gc.isenabled()
    gc.disable()

    for iteration in range(iterations):
        cam_idx = iteration % len(train_data)
        td = train_data[cam_idx]
        gt = td["image"]
        K_cam = td["K"]
        ext_cam = td["ext"]
        H, W = td["H"], td["W"]

        # Render using simple differentiable renderer
        opt_size = 64  # render size for optimization
        rendered = render_simple(
            means=means,
            quats=quats,
            log_scales=log_scales,
            logit_opacities=logit_opas,
            colors=raw_colors,
            viewmat=ext_cam,
            K=K_cam,
            width=opt_size,
            height=opt_size,
            max_gaussians=10_000,
        )

        # Downsample GT to match render size
        if H != opt_size or W != opt_size:
            gt_down = torch.nn.functional.interpolate(
                gt[:H, :W, :3].permute(2, 0, 1).unsqueeze(0),
                size=(opt_size, opt_size),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        else:
            gt_down = gt[:H, :W, :3]

        # Loss: L1 + lambda * (1 - SSIM)
        l1 = torch.abs(rendered - gt_down).mean()
        dssim = _ssim_loss(rendered, gt_down)
        loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim

        optimizer.zero_grad()
        try:
            loss.backward()
        except RuntimeError as exc:
            if "does not require grad" in str(exc):
                logger.warning("%s [OPT] Backward failed (%s), skipping optimization", TAG, exc)
                break
            else:
                raise

        last_loss = loss.item()

        # Accumulate gradients for densification
        if means.grad is not None:
            with torch.no_grad():
                grad_norms = means.grad.detach().norm(dim=-1)
                n_curr = min(len(grad_accum), len(grad_norms))
                grad_accum[:n_curr] += grad_norms[:n_curr]
                grad_count[:n_curr] += 1

        optimizer.step()

        # Clamp scales (use .data to avoid corrupting autograd state)
        with torch.no_grad():
            log_scales.data.clamp_(
                math.log(GEOMETRY_CONFIG.min_scale),
                math.log(GEOMETRY_CONFIG.max_scale),
            )

        # ── Densification ────────────────────────────────────────────────
        if (iteration + 1) % densify_interval == 0 and iteration < densify_until_iter:
            with torch.no_grad():
                avg_grad = grad_accum / (grad_count + 1e-8)
                dense_mask = avg_grad > gradient_threshold
                n_dense = int(dense_mask.sum())

                if n_dense > 0 and n_dense < n_gaussians // 2:
                    dense_idx = torch.where(dense_mask)[0]
                    curr_scales = torch.exp(log_scales[dense_idx])
                    scale_thresh = float(curr_scales.mean()) * 1.6

                    # Clone small high-gradient Gaussians
                    small_mask = curr_scales.max(dim=-1).values < scale_thresh
                    n_clone = int(small_mask.sum())

                    if n_clone > 0:
                        clone_idx = dense_idx[small_mask]
                        # Clone: copy with small position offset
                        # Use torch.cat on detached tensors, then requires_grad_ to make leaf
                        means = torch.cat([means.detach(), means[clone_idx].detach()]).requires_grad_(True)
                        log_scales = torch.cat([log_scales.detach(), log_scales[clone_idx].detach()]).requires_grad_(True)
                        quats = torch.cat([quats.detach(), quats[clone_idx].detach()]).requires_grad_(True)
                        logit_opas = torch.cat([logit_opas.detach(), logit_opas[clone_idx].detach()]).requires_grad_(True)
                        raw_colors = torch.cat([raw_colors.detach(), raw_colors[clone_idx].detach()]).requires_grad_(True)

                        n_gaussians = len(means)
                        grad_accum = torch.zeros(n_gaussians, device=device)
                        grad_count = torch.zeros(n_gaussians, device=device)
                        optimizer = make_optimizer()
                        logger.info("%s [OPT] Iter %d: cloned %d → %d Gaussians",
                                    TAG, iteration + 1, n_clone, n_gaussians)

                    # Split large high-gradient Gaussians
                    large_mask = curr_scales.max(dim=-1).values >= scale_thresh
                    n_split = int(large_mask.sum())

                    if n_split > 0:
                        split_idx = dense_idx[large_mask]
                        # Split: create 2 copies with halved scale
                        means = torch.cat([means.detach(), means[split_idx].detach()]).requires_grad_(True)
                        log_scales = torch.cat([log_scales.detach(), (log_scales[split_idx] - 0.3).detach()]).requires_grad_(True)
                        quats = torch.cat([quats.detach(), quats[split_idx].detach()]).requires_grad_(True)
                        logit_opas = torch.cat([logit_opas.detach(), logit_opas[split_idx].detach()]).requires_grad_(True)
                        raw_colors = torch.cat([raw_colors.detach(), raw_colors[split_idx].detach()]).requires_grad_(True)

                        n_gaussians = len(means)
                        grad_accum = torch.zeros(n_gaussians, device=device)
                        grad_count = torch.zeros(n_gaussians, device=device)
                        optimizer = make_optimizer()
                        logger.info("%s [OPT] Iter %d: split %d → %d Gaussians",
                                    TAG, iteration + 1, n_split, n_gaussians)

        # ── Pruning ──────────────────────────────────────────────────────
        if (iteration + 1) % 100 == 0 and iteration > 200:
            with torch.no_grad():
                opas_val = torch.sigmoid(logit_opas)
                keep = opas_val > prune_opacity
                if keep.sum() < len(keep) and keep.sum() > 100:
                    means = means[keep].detach().clone().requires_grad_(True)
                    log_scales = log_scales[keep].detach().clone().requires_grad_(True)
                    quats = quats[keep].detach().clone().requires_grad_(True)
                    logit_opas = logit_opas[keep].detach().clone().requires_grad_(True)
                    raw_colors = raw_colors[keep].detach().clone().requires_grad_(True)
                    n_gaussians = len(means)
                    grad_accum = torch.zeros(n_gaussians, device=device)
                    grad_count = torch.zeros(n_gaussians, device=device)
                    optimizer = make_optimizer()

        # Logging
        if (iteration + 1) % 50 == 0:
            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
            logger.info("%s [OPT] iter %d/%d: loss=%.4f, %d Gaussians",
                        TAG, iteration + 1, iterations, loss_val, n_gaussians)

    # Re-enable GC
    if gc_enabled:
        gc.enable()

    # ── Extract final parameters ─────────────────────────────────────────
    with torch.no_grad():
        final_positions = means.detach().cpu().numpy().astype(np.float32)
        final_colors = torch.sigmoid(raw_colors).detach().cpu().numpy().astype(np.float32)
        final_opacities = torch.sigmoid(logit_opas).detach().cpu().numpy().astype(np.float32)
        final_scales = torch.exp(log_scales).detach().cpu().numpy().astype(np.float32)
        final_quats = quats.detach().cpu().numpy().astype(np.float32)

        # Normalize quaternions
        qnorms = np.linalg.norm(final_quats, axis=1, keepdims=True)
        qnorms[qnorms < 1e-8] = 1e-8
        final_quats = final_quats / qnorms

        # Log-scale for splat_compiler
        final_scales = np.log(np.clip(final_scales, 1e-6, 0.05))

        if final_opacities.ndim == 1:
            final_opacities = final_opacities[:, None]

    opt_time = time.time() - t0
    logger.info("%s [OPT] Optimization complete: %d → %d Gaussians, loss=%.4f, %.1fs",
                TAG, gaussians.count, len(final_positions), last_loss, opt_time)

    return GaussianData(
        positions=final_positions,
        scales=final_scales,
        rotations=final_quats,
        colors=final_colors,
        opacities=final_opacities,
        errors=gaussians.errors,
    )


def unload():
    """Unload all reconstruction models and free VRAM."""
    global _DEPTH_MODEL

    if _DEPTH_MODEL is not None:
        del _DEPTH_MODEL
        _DEPTH_MODEL = None

    try:
        from sharp_wrapper import unload as unload_sharp

        unload_sharp()
    except Exception:
        pass

    if _has_cuda():
        import torch
        torch.cuda.empty_cache()

    logger.info("%s Reconstruction models unloaded", TAG)
