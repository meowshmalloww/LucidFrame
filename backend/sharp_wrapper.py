"""Apple SHARP adapter for high-fidelity single-image Gaussian reconstruction.

SHARP predicts metric, anisotropic 3D Gaussians directly from one photograph at
an internal 1536x1536 resolution.  The released checkpoint is licensed for
non-commercial research use; LucidFrame exposes that provenance in its UI and
manifests instead of presenting the model as an unrestricted dependency.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from reconstruction_stage import GaussianData

logger = logging.getLogger(__name__)
TAG = "[SHARP]"
SH_C0 = 0.28209479177387814

_PREDICTOR = None
_PREDICTOR_DEVICE = None
_MODEL_LOCK = threading.Lock()


def is_available() -> bool:
    """Return whether the optional SHARP package and CUDA are usable."""
    try:
        import sharp  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _load_predictor(device):
    global _PREDICTOR, _PREDICTOR_DEVICE
    with _MODEL_LOCK:
        if _PREDICTOR is not None and _PREDICTOR_DEVICE == str(device):
            return _PREDICTOR

        import torch
        from sharp.cli.predict import DEFAULT_MODEL_URL
        from sharp.models import PredictorParams, create_predictor

        started = time.time()
        logger.info("%s Loading official Apple checkpoint", TAG)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval().to(device)
        _PREDICTOR = predictor
        _PREDICTOR_DEVICE = str(device)
        logger.info("%s Predictor ready in %.1fs", TAG, time.time() - started)
        return predictor


def _linear_to_srgb(colors: np.ndarray) -> np.ndarray:
    colors = np.clip(colors, 0.0, 1.0)
    return np.where(
        colors <= 0.0031308,
        colors * 12.92,
        1.055 * np.power(colors, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def reconstruct_sharp(image_path: str | Path, output_dir: Path) -> GaussianData:
    """Predict a metric 3DGS scene from one image using SHARP."""
    if not is_available():
        raise RuntimeError("SHARP is not installed or CUDA is unavailable")

    import torch
    from sharp.cli.predict import predict_image
    from sharp.utils.io import load_rgb

    started = time.time()
    device = torch.device("cuda")
    predictor = _load_predictor(device)
    image, _, focal_px = load_rgb(Path(image_path))

    with _MODEL_LOCK, torch.inference_mode():
        predicted = predict_image(predictor, image, focal_px, device)

    positions = predicted.mean_vectors[0].detach().float().cpu().numpy()
    singular_values = predicted.singular_values[0].detach().float().cpu().numpy()
    rotations = predicted.quaternions[0].detach().float().cpu().numpy()
    colors = _linear_to_srgb(predicted.colors[0].detach().float().cpu().numpy())
    opacities = predicted.opacities[0].detach().float().cpu().numpy().reshape(-1, 1)

    min_opacity = float(os.getenv("SHARP_MIN_OPACITY", "0.01"))
    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(singular_values).all(axis=1)
        & np.isfinite(rotations).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(opacities[:, 0])
    )
    valid = finite & (positions[:, 2] > 0.01) & (opacities[:, 0] >= min_opacity)
    if int(valid.sum()) < 1000:
        raise RuntimeError("SHARP produced too few valid Gaussians")

    positions = positions[valid].astype(np.float32)
    singular_values = np.clip(singular_values[valid], 1e-6, None).astype(np.float32)
    rotations = rotations[valid].astype(np.float32)  # SHARP uses WXYZ.
    colors = colors[valid].astype(np.float32)
    opacities = opacities[valid].astype(np.float32)

    result = GaussianData(
        positions=positions,
        scales=np.log(singular_values).astype(np.float32),
        rotations=rotations,
        colors=colors,
        opacities=opacities,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    linear_scales = singular_values.reshape(-1)
    report = {
        "backend": "Apple SHARP",
        "license": "Apple Machine Learning Research Model License (non-commercial research)",
        "source_resolution": [int(image.shape[1]), int(image.shape[0])],
        "inference_resolution": [1536, 1536],
        "focal_px": round(float(focal_px), 3),
        "gaussian_count": result.count,
        "filtered_count": int(len(valid) - valid.sum()),
        "median_depth_m": round(float(np.median(positions[:, 2])), 5),
        "scale_percentiles": [
            round(float(value), 7)
            for value in np.quantile(linear_scales, [0.01, 0.5, 0.99])
        ],
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "sharp_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%s Produced %d Gaussians in %.1fs", TAG, result.count, time.time() - started)
    return result


def gaussian_data_from_ply(path: str | Path, *, y_up: bool) -> GaussianData:
    """Read a standard 3DGS PLY and convert it to LucidFrame's binary contract."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    scales = np.stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
    ).astype(np.float32)
    rotations = np.stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1
    ).astype(np.float32)
    colors = np.stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1
    ).astype(np.float32)
    colors = np.clip(colors * SH_C0 + 0.5, 0.0, 1.0)
    opacity_logits = np.asarray(vertex["opacity"], dtype=np.float32)
    opacities = (1.0 / (1.0 + np.exp(-np.clip(opacity_logits, -30.0, 30.0))))[:, None]

    if y_up:
        # Reflect both centres and covariance bases into gsplat.js camera axes:
        # X right, Y down, Z forward.  M*R is a reflection, so a second axis
        # flip F restores a proper rotation without changing the covariance.
        positions[:, 1] *= -1.0
        matrices = _wxyz_to_matrices(rotations)
        mirror = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
        parity = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
        matrices = mirror[None] @ matrices @ parity[None]
        rotations = _matrices_to_wxyz(matrices)

    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(rotations).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(opacities[:, 0])
    )
    return GaussianData(
        positions=positions[finite],
        scales=scales[finite],
        rotations=rotations[finite],
        colors=colors[finite],
        opacities=opacities[finite].astype(np.float32),
    )


def _wxyz_to_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-8)
    w, x, y, z = quaternions.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3).astype(np.float32)


def _matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
    from panorama_spherical_stage import _rotation_matrices_to_wxyz

    return _rotation_matrices_to_wxyz(matrices)


def unload() -> None:
    """Release the cached predictor and its VRAM."""
    global _PREDICTOR, _PREDICTOR_DEVICE
    with _MODEL_LOCK:
        _PREDICTOR = None
        _PREDICTOR_DEVICE = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
