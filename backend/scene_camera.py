"""Camera calibration metadata for SHARP reconstructions.

SHARP predicts metric Gaussians using the focal length supplied at inference.
The web viewer must use the same pinhole camera, otherwise even correct 3D
centres reproject to different pixels and appear to slide during movement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def camera_metadata_from_quality(report: dict[str, Any]) -> dict[str, float] | None:
    """Convert one ``sharp_quality.json`` report into viewer-safe metadata."""
    resolution = report.get("source_resolution")
    focal_px = report.get("focal_px")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(isinstance(value, (int, float)) for value in resolution)
        or not isinstance(focal_px, (int, float))
    ):
        return None

    width, height = float(resolution[0]), float(resolution[1])
    focal = float(focal_px)
    if width <= 0 or height <= 0 or focal <= 0 or not math.isfinite(focal):
        return None

    metadata: dict[str, float] = {
        "horizontal_fov_deg": round(math.degrees(2.0 * math.atan(width / (2.0 * focal))), 5),
        "focal_px": round(focal, 5),
    }

    depth_percentiles = report.get("depth_percentiles_m")
    if isinstance(depth_percentiles, list) and depth_percentiles:
        near_depth = depth_percentiles[0]
        if isinstance(near_depth, (int, float)) and float(near_depth) > 0:
            near_depth_m = float(near_depth)
            normalized_diagonal = math.hypot(width / focal, height / focal)
            lateral_m = 0.08 * normalized_diagonal * near_depth_m
            forward_m = 0.15 * near_depth_m
            # Cross roughly half of SHARP's official nearby-view envelope per
            # second. There is deliberately no invisible collision boundary.
            speed_mps = min(max(min(lateral_m, forward_m) * 0.5, 0.08), 0.55)
            metadata.update(
                {
                    "near_depth_m": round(near_depth_m, 5),
                    "recommended_lateral_m": round(lateral_m, 5),
                    "recommended_forward_m": round(forward_m, 5),
                    "move_speed_mps": round(speed_mps, 5),
                }
            )
    else:
        # Older saved scenes still benefit from calibrated projection. Use a
        # conservative speed until they are regenerated with depth quantiles.
        metadata["move_speed_mps"] = 0.12

    return metadata


def load_scene_camera(output_dir: str | Path) -> dict[str, float] | None:
    """Read SHARP quality data for a generated scene, if present and valid."""
    report_path = Path(output_dir) / "sharp_quality.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    return camera_metadata_from_quality(report)
