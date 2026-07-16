"""
Splat Compiler — Convert Gaussian parameters to binary .splat format.

The .splat format is a flat binary array supported by the browser renderer.
Each Gaussian occupies 32 bytes:
  - Position: x, y, z (float32 × 3 = 12 bytes)
  - Scale: sx, sy, sz (float32 × 3 = 12 bytes)
  - Color: r, g, b, a (uint8 × 4 = 4 bytes)
  - Rotation: w, x, y, z (uint8 × 4 = 4 bytes, normalized from float)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from reconstruction_stage import GaussianData

logger = logging.getLogger(__name__)
TAG = "[SPLAT_COMPILER]"


def compile_splat(gaussians: GaussianData, output_path: Path) -> Path:
    """
    Convert GaussianData to binary .splat format and save.

    Args:
        gaussians: GaussianData with positions, scales, rotations, colors, opacities.
        output_path: Where to save the .splat file.

    Returns:
        Path to the saved .splat file.
    """
    if gaussians.count == 0:
        raise ValueError("No Gaussians to compile")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = gaussians.count
    logger.info("%s Compiling %d Gaussians to .splat format...", TAG, n)

    # Process positions (already float32)
    positions = gaussians.positions.astype(np.float32)

    # Process scales: exponentiate (models output log-scale) → clamp → float32
    scales = np.exp(gaussians.scales.astype(np.float32))  # undo log-scale
    # Preserve the learned anisotropic support. SHARP legitimately predicts
    # broad wall/background axes (the released model reaches roughly 0.85 m on
    # our regression scene); the former 10 cm ceiling flattened those surfaces
    # and opened pinholes in the web render. Keep only an outlier guard here.
    max_scale = float(os.getenv("SPLAT_MAX_SCALE", "2.00"))
    max_scale = min(max(max_scale, 0.02), 2.00)
    clipped = int(np.count_nonzero((scales < 0.00005) | (scales > max_scale)))
    scales = np.clip(scales, 0.00005, max_scale)
    if clipped:
        logger.info(
            "%s Clamped %d/%d scale axes outside [0.00005, %.3f]",
            TAG,
            clipped,
            scales.size,
            max_scale,
        )

    # Process colors: clamp to [0, 255] → uint8
    colors = gaussians.colors.astype(np.float32)
    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)

    # Process opacities: sigmoid → [0, 255] → uint8
    opacities = gaussians.opacities.astype(np.float32)
    opacities = np.clip(opacities * 255, 0, 255).astype(np.uint8)

    # Process rotations: normalize quaternion → map [-1,1] to [0,255] → uint8
    rotations = gaussians.rotations.astype(np.float32)
    # Normalize quaternions
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    rotations = rotations / norms
    # Map from [-1, 1] to the standard .splat signed-byte encoding.
    rotations_u8 = np.clip(rotations * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # Combine RGBA
    rgba = np.column_stack([colors, opacities.flatten()])  # (N, 4) uint8

    # Build binary buffer via structured array (vectorized — no Python loop)
    # Each Gaussian: 32 bytes = 6 float32 (pos + scale) + 4 uint8 (rgba) + 4 uint8 (rot)
    dtype = np.dtype([
        ('pos_scale', np.float32, 6),   # 24 bytes
        ('rgba', np.uint8, 4),           # 4 bytes
        ('rot', np.uint8, 4),            # 4 bytes
    ])
    buf = np.empty(n, dtype=dtype)
    buf['pos_scale'] = np.column_stack([positions, scales])  # (N, 6)
    buf['rgba'] = rgba
    buf['rot'] = rotations_u8

    with open(output_path, "wb") as f:
        f.write(buf.tobytes())

    file_size = output_path.stat().st_size
    logger.info("%s Compiled %d Gaussians → %s (%.2f MB)", TAG, n, output_path, file_size / 1e6)

    return output_path
