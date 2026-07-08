"""
Splat Compiler — Convert Gaussian parameters to binary .splat format.

The .splat format is a flat binary array used by gsplat.js.
Each Gaussian occupies 32 bytes:
  - Position: x, y, z (float32 × 3 = 12 bytes)
  - Scale: sx, sy, sz (float32 × 3 = 12 bytes)
  - Color: r, g, b, a (uint8 × 4 = 4 bytes)
  - Rotation: w, x, y, z (uint8 × 4 = 4 bytes, normalized from float)
"""
from __future__ import annotations

import logging
import struct
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
    scales = np.clip(scales, 0.001, 0.5)  # clamp to reasonable range

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
    # Map from [-1, 1] to [0, 255]
    rotations_u8 = np.clip((rotations + 1.0) * 127.5, 0, 255).astype(np.uint8)

    # Combine RGBA
    rgba = np.column_stack([colors, opacities.flatten()])  # (N, 4) uint8

    # Write binary
    with open(output_path, "wb") as f:
        for i in range(n):
            # Position: 12 bytes (3 × float32)
            f.write(struct.pack("fff", positions[i, 0], positions[i, 1], positions[i, 2]))
            # Scale: 12 bytes (3 × float32)
            f.write(struct.pack("fff", scales[i, 0], scales[i, 1], scales[i, 2]))
            # Color: 4 bytes (4 × uint8)
            f.write(struct.pack("BBBB", rgba[i, 0], rgba[i, 1], rgba[i, 2], rgba[i, 3]))
            # Rotation: 4 bytes (4 × uint8)
            f.write(struct.pack("BBBB", rotations_u8[i, 0], rotations_u8[i, 1], rotations_u8[i, 2], rotations_u8[i, 3]))

    file_size = output_path.stat().st_size
    logger.info("%s Compiled %d Gaussians → %s (%.2f MB)", TAG, n, output_path, file_size / 1e6)

    return output_path
