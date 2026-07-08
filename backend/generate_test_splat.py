"""
Generate a test .splat file with colorful Gaussian points arranged in a sphere.

This lets us verify the frontend gsplat.js viewer works without running the
full ML pipeline. The .splat format is 32 bytes per Gaussian:
  - Position: 3x float32 (12 bytes)
  - Scale: 3x float32 (12 bytes)
  - Color: 4x uint8 (4 bytes)
  - Rotation: 4x uint8 (4 bytes, quaternion normalized from [-1,1])

Usage:
    python generate_test_splat.py [output_path] [num_gaussians]
"""
import math
import os
import struct
import sys
import random

import numpy as np


def generate_sphere_splat(num_gaussians: int = 50000) -> bytes:
    """Generate Gaussians arranged on a sphere surface with rainbow colors."""
    # Fibonacci sphere distribution for even coverage
    indices = np.arange(0, num_gaussians, dtype=np.float32)
    phi = np.arccos(1 - 2 * indices / num_gaussians)
    theta = math.pi * (1 + 5**0.5) * indices

    # Sphere radius with some noise for depth
    radius = 2.0 + np.random.randn(num_gaussians).astype(np.float32) * 0.1

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Small uniform scales
    sx = np.full(num_gaussians, 0.02, dtype=np.float32)
    sy = np.full(num_gaussians, 0.02, dtype=np.float32)
    sz = np.full(num_gaussians, 0.02, dtype=np.float32)

    # Rainbow colors based on position
    r = ((np.sin(theta) + 1) / 2 * 255).astype(np.uint8)
    g = ((np.cos(phi) + 1) / 2 * 255).astype(np.uint8)
    b = ((np.sin(theta + phi) + 1) / 2 * 255).astype(np.uint8)
    a = np.full(num_gaussians, 255, dtype=np.uint8)

    # Identity rotation: w=1, x=0, y=0, z=0
    # Map from [-1,1] to [0,255]: w=1 -> 255, x=0 -> 128, etc.
    qw = np.full(num_gaussians, 255, dtype=np.uint8)
    qx = np.full(num_gaussians, 128, dtype=np.uint8)
    qy = np.full(num_gaussians, 128, dtype=np.uint8)
    qz = np.full(num_gaussians, 128, dtype=np.uint8)

    # Pack into binary: 32 bytes per Gaussian
    data = bytearray()
    for i in range(num_gaussians):
        data += struct.pack(
            "<fff fff BBBB BBBB",
            float(x[i]), float(y[i]), float(z[i]),
            float(sx[i]), float(sy[i]), float(sz[i]),
            int(r[i]), int(g[i]), int(b[i]), int(a[i]),
            int(qw[i]), int(qx[i]), int(qy[i]), int(qz[i]),
        )

    return bytes(data)


def generate_galaxy_splat(num_gaussians: int = 80000) -> bytes:
    """Generate a spiral galaxy pattern — more visually interesting."""
    # Spiral arms
    arms = 3
    indices = np.arange(0, num_gaussians, dtype=np.float32)
    arm = indices % arms
    t = indices / num_gaussians

    angle = t * math.pi * 4 + (arm / arms) * 2 * math.pi
    radius = t * 3.0 + np.random.randn(num_gaussians).astype(np.float32) * 0.15

    x = radius * np.cos(angle) + np.random.randn(num_gaussians).astype(np.float32) * 0.1
    y = (np.random.randn(num_gaussians).astype(np.float32) * 0.3) * (1 - t)  # flatten at edges
    z = radius * np.sin(angle) + np.random.randn(num_gaussians).astype(np.float32) * 0.1

    # Color: hot core (white/yellow) to cool edges (blue/purple)
    heat = 1 - t
    r = (255 * (0.5 + heat * 0.5)).astype(np.uint8)
    g = (255 * (0.3 + heat * 0.7)).astype(np.uint8)
    b = (255 * (0.8 + (1 - heat) * 0.2)).astype(np.uint8)
    a = np.full(num_gaussians, 255, dtype=np.uint8)

    # Varying scale — smaller at core, larger at edges
    scale = 0.01 + t * 0.03
    sx = scale.astype(np.float32)
    sy = scale.astype(np.float32)
    sz = scale.astype(np.float32)

    # Identity rotation
    qw = np.full(num_gaussians, 255, dtype=np.uint8)
    qx = np.full(num_gaussians, 128, dtype=np.uint8)
    qy = np.full(num_gaussians, 128, dtype=np.uint8)
    qz = np.full(num_gaussians, 128, dtype=np.uint8)

    data = bytearray()
    for i in range(num_gaussians):
        data += struct.pack(
            "<fff fff BBBB BBBB",
            float(x[i]), float(y[i]), float(z[i]),
            float(sx[i]), float(sy[i]), float(sz[i]),
            int(r[i]), int(g[i]), int(b[i]), int(a[i]),
            int(qw[i]), int(qx[i]), int(qy[i]), int(qz[i]),
        )

    return bytes(data)


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/test_galaxy.splat"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 50000

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Generating galaxy .splat with {num} Gaussians...")
    data = generate_galaxy_splat(num)

    with open(output_path, "wb") as f:
        f.write(data)

    size_kb = len(data) / 1024
    print(f"Written: {output_path} ({size_kb:.1f} KB, {num} Gaussians)")
    print(f"Expected size: {num * 32} bytes = {num * 32 / 1024:.1f} KB")
    print("OK")


if __name__ == "__main__":
    main()
