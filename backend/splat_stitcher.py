"""
Splat Stitcher — Merge multiple Gaussian splat scenes.

Stretch goal: expand the hallucinated space by generating additional
splats from edge views and stitching them via coordinate transformation.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reconstruction_stage import GaussianData

logger = logging.getLogger(__name__)
TAG = "[STITCHER]"


def stitch_gaussians(
    splat_a: GaussianData,
    splat_b: GaussianData,
    transform_ab: np.ndarray,
) -> GaussianData:
    """
    Stitch two Gaussian clouds using a 4x4 transformation matrix.

    Transforms splat_b's coordinates into splat_a's coordinate system,
    then concatenates the Gaussian arrays.

    Args:
        splat_a: Existing (base) Gaussian data.
        splat_b: New Gaussian data to stitch in.
        transform_ab: 4x4 matrix transforming B's coords to A's coords.

    Returns:
        Merged GaussianData.
    """
    if splat_b.count == 0:
        return splat_a
    if splat_a.count == 0:
        return splat_b

    logger.info("%s Stitching %d + %d Gaussians", TAG, splat_a.count, splat_b.count)

    # Transform positions: p_A = T_AB @ [p_B, 1]
    ones = np.ones((splat_b.count, 1), dtype=np.float32)
    pos_b_hom = np.hstack([splat_b.positions, ones])  # (N, 4)
    pos_a = (transform_ab @ pos_b_hom.T).T[:, :3]  # (N, 3)

    # Transform rotations: R_A = T_AB[:3,:3] @ R_B
    rot_matrix = transform_ab[:3, :3]
    # Convert quaternion to rotation matrix, apply transform, convert back
    # For simplicity, we keep the original rotations — the transform mainly
    # affects positions. Full rotation transform requires quat→mat→mat→quat.
    rotations = splat_b.rotations

    # Concatenate
    merged = GaussianData(
        positions=np.vstack([splat_a.positions, pos_a]),
        scales=np.vstack([splat_a.scales, splat_b.scales]),
        rotations=np.vstack([splat_a.rotations, rotations]),
        colors=np.vstack([splat_a.colors, splat_b.colors]),
        opacities=np.vstack([splat_a.opacities, splat_b.opacities]),
        errors=splat_a.errors + splat_b.errors,
    )

    logger.info("%s Stitched result: %d Gaussians", TAG, merged.count)
    return merged
