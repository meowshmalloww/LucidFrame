"""High-detail local panorama reconstruction through SPAG4D SHARP-360.

Unlike a depth-projected spherical shell, this backend asks SHARP to predict
full anisotropic Gaussians for six overlapping horizon views plus zenith and
nadir caps.  LucidFrame's aligned panoramic depth is used to bring every face
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
    _normalise_panorama,
    estimate_aligned_panorama_disparity,
    validate_panorama,
)
from reconstruction_stage import GaussianData
from sharp_wrapper import gaussian_data_from_ply, is_available

logger = logging.getLogger(__name__)
TAG = "[SHARP360]"
_CONVERSION_LOCK = threading.Lock()


def reconstruct_sharp360(image_path: str | Path, output_dir: Path) -> GaussianData:
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
    target_width = int(os.getenv("SHARP360_ERP_WIDTH", "1536"))
    target_width = max(1024, min(3072, int(round(target_width / 32)) * 32))
    panorama, observed_mask, profile = _normalise_panorama(source, target_width)
    normalized_path = output_dir / "sharp360_normalised_panorama.png"
    Image.fromarray(panorama).save(normalized_path)

    side_count = int(os.getenv("SHARP360_SIDE_COUNT", "4"))
    side_count = side_count if side_count in {4, 6, 8, 10, 12} else 6
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
        spag_sharp360.predict_da360_disparity = estimate_aligned_panorama_disparity
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

    report = {
        **profile,
        "backend": "SPAG4D SHARP-360 with LucidFrame aligned depth",
        "license": "SPAG4D MIT; Apple SHARP model is non-commercial research",
        "gaussian_count": result.count,
        "predicted_faces": int(stats.get("num_faces", side_count + (2 if include_caps else 0))),
        "horizon_faces": side_count,
        "pole_caps": include_caps,
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
