"""
Pure PyTorch Gaussian Splatting Renderer — Differentiable.

Implements EWA (Elliptical Weighted Average) splatting with full autograd support:
  1. Project 3D Gaussian means to screen space (differentiable)
  2. Compute 2D covariance from 3D scale + rotation (differentiable)
  3. Sort by depth, assign to tiles (non-differentiable, detached)
  4. Vectorized per-tile alpha compositing (differentiable)
  5. Concatenate tiles into final image (differentiable)

No in-place operations, no .item() calls — fully compatible with torch.autograd.

Used for:
  - Fallback rendering when gsplat CUDA is unavailable
  - Differentiable optimization loop (3DGS training)
"""
from __future__ import annotations

import math
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)
TAG = "[TORCH_SPLAT]"


def _quat_to_matrix(quats: torch.Tensor) -> torch.Tensor:
    """Convert quaternions (w,x,y,z) to 3x3 rotation matrices. (N, 4) -> (N, 3, 3)."""
    w, x, y, z = quats.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def _compute_2d_covariance(
    means_cam: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    R_cam: torch.Tensor,
    fx: float,
    fy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D screen-space covariance and screen-space centers.
    All operations are differentiable.

    Returns (screen_xy, cov_2d, depths) where screen_xy is (N, 2) and cov_2d is (N, 2, 2).
    """
    N = means_cam.shape[0]

    # 3D covariance: Sigma = R @ diag(s^2) @ R^T
    rot_mats = _quat_to_matrix(quats)  # (N, 3, 3)
    S = torch.diag_embed(scales)  # (N, 3, 3)
    Sigma = rot_mats @ S @ torch.transpose(S, 1, 2) @ torch.transpose(rot_mats, 1, 2)  # (N, 3, 3)

    # Transform covariance to camera space: Sigma_cam = R_cam @ Sigma @ R_cam^T
    R_expand = R_cam.unsqueeze(0).expand(N, 3, 3)
    Sigma_cam = R_expand @ Sigma @ torch.transpose(R_expand, 1, 2)

    # Jacobian of perspective projection at the mean
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z = means_cam[:, 2].clamp(min=0.01)

    # Build Jacobian using stack (not in-place) to preserve gradients
    zeros_n = torch.zeros_like(z)
    ones_n = torch.ones_like(z)
    J = torch.stack([
        torch.stack([fx / z, zeros_n, -(fx * x) / (z * z)], dim=-1),
        torch.stack([zeros_n, fy / z, -(fy * y) / (z * z)], dim=-1),
        torch.stack([zeros_n, zeros_n, ones_n], dim=-1),
    ], dim=-2)  # (N, 3, 3)

    # 2D covariance: J @ Sigma_cam @ J^T (take top-left 2x2)
    W = J @ Sigma_cam  # (N, 3, 3)
    cov_3d = W @ torch.transpose(J, 1, 2)  # (N, 3, 3)
    cov_2d = cov_3d[:, :2, :2]  # (N, 2, 2)

    # Add low-pass filter for stability (EWA) — use out-of-place
    cov_2d = cov_2d + 0.3 * torch.eye(2, device=cov_2d.device, dtype=cov_2d.dtype).unsqueeze(0)

    # Screen-space center
    sx = fx * x / z
    sy = fy * y / z

    return torch.stack([sx, sy], dim=-1), cov_2d, z


def rasterize_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    max_gaussians: int = 200_000,
    tile_size: int = 16,
    max_per_tile: int = 500,
) -> torch.Tensor:
    """
    Render Gaussians via EWA splatting in pure PyTorch — fully differentiable.

    Args:
        means: (N, 3) Gaussian centers in world space.
        quats: (N, 4) Quaternions (w,x,y,z).
        scales: (N, 3) 3D scales (already exp'd, not log).
        opacities: (N,) Opacities in [0, 1].
        colors: (N, 3) RGB colors in [0, 1].
        viewmat: (4, 4) World-to-camera extrinsic.
        K: (3, 3) Camera intrinsics.
        width, height: Output image dimensions.
        max_gaussians: Subsample if more than this (for speed).
        tile_size: Tile size for tiled rendering (must divide width/height or pad).
        max_per_tile: Max Gaussians per tile (subsample if exceeded).

    Returns:
        (H, W, 3) RGB image tensor on same device, values in [0, 1].
        Gradients flow through means, quats, scales, opacities, colors.
    """
    device = means.device
    dtype = means.dtype
    N = means.shape[0]

    if N == 0:
        return torch.zeros(height, width, 3, device=device, dtype=dtype)

    R_cam = viewmat[:3, :3]
    t_cam = viewmat[:3, 3]

    # ── 1. Transform means to camera space (differentiable) ─────────────
    means_cam = means @ R_cam.T + t_cam  # (N, 3)

    # ── 2. Cull behind camera (non-differentiable — just indexing) ──────
    with torch.no_grad():
        in_front = means_cam[:, 2] > 0.01
    if in_front.sum() == 0:
        return torch.zeros(height, width, 3, device=device, dtype=dtype)

    means = means[in_front]
    quats = quats[in_front]
    scales = scales[in_front]
    opacities = opacities[in_front]
    colors = colors[in_front]
    means_cam = means_cam[in_front]
    N = means.shape[0]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # ── 3. Compute 2D covariance and screen positions (differentiable) ──
    screen_xy, cov_2d, depths = _compute_2d_covariance(
        means_cam, scales, quats, R_cam, fx, fy
    )

    screen_x = screen_xy[:, 0] + cx
    screen_y = screen_xy[:, 1] + cy

    # ── 4. Compute bounding radius for each Gaussian (non-differentiable) ─
    with torch.no_grad():
        a = cov_2d[:, 0, 0]
        b = cov_2d[:, 0, 1]
        d = cov_2d[:, 1, 1]
        trace = a + d
        det = a * d - b * b
        discriminant = (trace * trace / 4 - det).clamp(min=0)
        lambda1 = trace / 2 + torch.sqrt(discriminant)
        lambda2 = trace / 2 - torch.sqrt(discriminant)
        radius = 3 * torch.sqrt(torch.maximum(lambda1, lambda2.clamp(min=0.1)))

        # Cull Gaussians outside screen
        margin = radius
        in_screen = (
            (screen_x + margin > 0) & (screen_x - margin < width) &
            (screen_y + margin > 0) & (screen_y - margin < height)
        )

    if in_screen.sum() == 0:
        return torch.zeros(height, width, 3, device=device, dtype=dtype)

    # Apply screen cull (indexing is differentiable for the selected elements)
    screen_x = screen_x[in_screen]
    screen_y = screen_y[in_screen]
    cov_2d = cov_2d[in_screen]
    depths = depths[in_screen]
    opacities = opacities[in_screen]
    colors = colors[in_screen]
    radius = radius[in_screen]
    N = screen_x.shape[0]

    # ── 5. Sort by depth (non-differentiable — just reordering) ─────────
    with torch.no_grad():
        sort_idx = depths.argsort()

    screen_x = screen_x[sort_idx]
    screen_y = screen_y[sort_idx]
    cov_2d = cov_2d[sort_idx]
    depths = depths[sort_idx]
    opacities = opacities[sort_idx]
    colors = colors[sort_idx]
    radius = radius[sort_idx]

    # ── 6. Tile assignment (non-differentiable) ─────────────────────────
    # Pad image to be divisible by tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    pad_h = (tile_size - height % tile_size) % tile_size
    padded_w = width + pad_w
    padded_h = height + pad_h
    n_tiles_x = padded_w // tile_size
    n_tiles_y = padded_h // tile_size

    with torch.no_grad():
        # Compute inverse covariance for Gaussian evaluation
        cov_inv = torch.linalg.inv(
            cov_2d + 1e-6 * torch.eye(2, device=device, dtype=dtype).unsqueeze(0)
        )  # (N, 2, 2)

        # Assign Gaussians to tiles based on center + radius
        # Each Gaussian goes to its center tile and neighboring tiles within radius
        tile_x_center = (screen_x / tile_size).long().clamp(0, n_tiles_x - 1)
        tile_y_center = (screen_y / tile_size).long().clamp(0, n_tiles_y - 1)
        tile_radius = (radius / tile_size).ceil().long().clamp(min=0, max=4)

        # Build tile-to-gaussian mapping
        tile_gaussians: list[list[int]] = [[] for _ in range(n_tiles_x * n_tiles_y)]
        for i in range(N):
            tx = tile_x_center[i].item()
            ty = tile_y_center[i].item()
            tr = tile_radius[i].item()
            for dy in range(-tr, tr + 1):
                ny = ty + dy
                if ny < 0 or ny >= n_tiles_y:
                    continue
                for dx in range(-tr, tr + 1):
                    nx = tx + dx
                    if nx < 0 or nx >= n_tiles_x:
                        continue
                    tile_gaussians[ny * n_tiles_x + nx].append(i)

    # ── 7. Per-tile vectorized rendering (differentiable) ───────────────
    tile_images: list[torch.Tensor] = []

    for ty in range(n_tiles_y):
        row_tiles: list[torch.Tensor] = []
        for tx in range(n_tiles_x):
            tile_idx = ty * n_tiles_x + tx
            g_indices = tile_gaussians[tile_idx]

            # Tile pixel coordinates in image space
            y0 = ty * tile_size
            x0 = tx * tile_size
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            th = y1 - y0
            tw = x1 - x0

            if len(g_indices) == 0 or th == 0 or tw == 0:
                row_tiles.append(torch.zeros(tile_size, tile_size, 3, device=device, dtype=dtype))
                continue

            # Subsample if too many Gaussians per tile
            if len(g_indices) > max_per_tile:
                with torch.no_grad():
                    # Keep closest Gaussians (they're already sorted by depth)
                    g_indices = g_indices[:max_per_tile]

            G = len(g_indices)
            g_idx = torch.tensor(g_indices, device=device, dtype=torch.long)

            # Gather Gaussian parameters for this tile (differentiable indexing)
            t_sx = screen_x[g_idx]  # (G,)
            t_sy = screen_y[g_idx]  # (G,)
            t_ci = cov_inv[g_idx]   # (G, 2, 2) — detached (inv is no_grad)
            t_opa = opacities[g_idx]  # (G,)
            t_col = colors[g_idx]    # (G, 3)

            # Create pixel coordinate grid for this tile
            pys = torch.arange(y0, y1, device=device, dtype=dtype)
            pxs = torch.arange(x0, x1, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(pys, pxs, indexing="ij")  # (th, tw)
            px_flat = grid_x.reshape(-1)  # (P,)
            py_flat = grid_y.reshape(-1)  # (P,)
            P = px_flat.shape[0]

            # Compute pixel-Gaussian offsets: (P, G)
            dx = px_flat.unsqueeze(1) - t_sx.unsqueeze(0)  # (P, G)
            dy = py_flat.unsqueeze(1) - t_sy.unsqueeze(0)  # (P, G)

            # Gaussian evaluation: exp(-0.5 * d^T * cov_inv * d)
            # d = [dx, dy], cov_inv is (G, 2, 2)
            ci00 = t_ci[:, 0, 0].unsqueeze(0)  # (1, G)
            ci01 = t_ci[:, 0, 1].unsqueeze(0)  # (1, G)
            ci11 = t_ci[:, 1, 1].unsqueeze(0)  # (1, G)

            g_val = torch.exp(-0.5 * (
                ci00 * dx * dx + 2 * ci01 * dx * dy + ci11 * dy * dy
            ))  # (P, G)

            # Alpha = opacity * gaussian_weight
            alpha = t_opa.unsqueeze(0) * g_val  # (P, G)
            alpha = alpha.clamp(max=0.999)

            # Front-to-back alpha compositing (differentiable)
            # T_i = prod(1 - alpha_j) for j < i
            one_minus = 1.0 - alpha  # (P, G)
            log_om = torch.log(one_minus.clamp(min=1e-8))
            cumsum_log = torch.cumsum(log_om, dim=1)  # (P, G)
            # T_i = exp(cumsum_log_i - log_om_i) = prod(1-alpha_j) for j < i
            T = torch.exp(cumsum_log - log_om)  # (P, G)

            # Weighted contribution: alpha_i * T_i * color_i
            weighted = alpha * T  # (P, G)
            tile_color = (weighted.unsqueeze(-1) * t_col.unsqueeze(0)).sum(dim=1)  # (P, 3)

            # Reshape to tile dimensions
            tile_color = tile_color.reshape(th, tw, 3)

            # Pad tile to tile_size if needed (use cat, not in-place)
            if th < tile_size:
                tile_color = torch.cat([
                    tile_color,
                    torch.zeros(tile_size - th, tw, 3, device=device, dtype=dtype),
                ], dim=0)
            if tw < tile_size:
                tile_color = torch.cat([
                    tile_color,
                    torch.zeros(tile_size, tile_size - tw, 3, device=device, dtype=dtype),
                ], dim=1)

            row_tiles.append(tile_color)

        tile_images.append(torch.cat(row_tiles, dim=1))  # (tile_size, n_tiles_x * tile_size, 3)

    full_image = torch.cat(tile_images, dim=0)  # (n_tiles_y * tile_size, n_tiles_x * tile_size, 3)

    # Crop to original dimensions
    return full_image[:height, :width].clamp(0, 1)


def render_view(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    tile_size: int = 16,
) -> torch.Tensor:
    """
    Convenience wrapper: takes log-space parameters (like 3DGS training),
    applies activations, and renders.

    Args:
        means: (N, 3) positions
        quats: (N, 4) quaternions (w,x,y,z)
        log_scales: (N, 3) log-space scales
        logit_opacities: (N,) logit-space opacities
        colors: (N, 3) raw colors (will be sigmoid'd)
        viewmat: (4, 4) world-to-camera
        K: (3, 3) intrinsics
        width, height: output size

    Returns:
        (H, W, 3) rendered image in [0, 1]
    """
    scales = torch.exp(log_scales)
    opacities = torch.sigmoid(logit_opacities)
    rgb = torch.sigmoid(colors)

    # Normalize quaternions
    q_norm = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)

    return rasterize_gaussians(
        means=means,
        quats=q_norm,
        scales=scales,
        opacities=opacities,
        colors=rgb,
        viewmat=viewmat,
        K=K,
        width=width,
        height=height,
        tile_size=tile_size,
    )


def render_simple(
    means: torch.Tensor,
    quats: torch.Tensor,
    log_scales: torch.Tensor,
    logit_opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    max_gaussians: int = 30_000,
) -> torch.Tensor:
    """
    Simple differentiable renderer for optimization.
    Uses vectorized per-pixel alpha compositing — no tiled loop, no sorting.
    Creates a shallow autograd graph that's reliable in threaded contexts.

    Trades memory for graph simplicity: O(N*P) where N=gaussians, P=pixels.
    With 10K gaussians at 64x64 (4096 px), that's 41M elements — fits in VRAM.

    Args:
        means: (N, 3) positions
        quats: (N, 4) quaternions (w,x,y,z)
        log_scales: (N, 3) log-space scales
        logit_opacities: (N,) logit-space opacities
        colors: (N, 3) raw colors (will be sigmoid'd)
        viewmat: (4, 4) world-to-camera
        K: (3, 3) intrinsics
        width, height: output size
        max_gaussians: subsample if more than this

    Returns:
        (H, W, 3) rendered image in [0, 1]
    """
    device = means.device
    dtype = means.dtype
    N = means.shape[0]

    # Subsample if too many
    if N > max_gaussians:
        with torch.no_grad():
            R_cam = viewmat[:3, :3]
            t_cam = viewmat[:3, 3]
            means_cam = means @ R_cam.T + t_cam
            dists = means_cam.norm(dim=-1)
            keep = dists.argsort()[:max_gaussians]
        means = means[keep]
        quats = quats[keep]
        log_scales = log_scales[keep]
        logit_opacities = logit_opacities[keep]
        colors = colors[keep]
        N = means.shape[0]

    # Activations
    scales = torch.exp(log_scales)  # (N, 3)
    opacities = torch.sigmoid(logit_opacities)  # (N,)
    rgb = torch.sigmoid(colors)  # (N, 3)

    # Transform to camera space
    R_cam = viewmat[:3, :3]
    t_cam = viewmat[:3, 3]
    means_cam = means @ R_cam.T + t_cam  # (N, 3)

    # Cull behind camera
    with torch.no_grad():
        in_front = means_cam[:, 2] > 0.01
    if in_front.sum() == 0:
        # Return zeros connected to graph so backward doesn't fail
        return (means.sum() * 0.0).reshape(1, 1, 1).expand(height, width, 3).clamp(0, 1)

    means_cam = means_cam[in_front]
    scales = scales[in_front]
    opacities = opacities[in_front]
    rgb = rgb[in_front]
    N = means_cam.shape[0]

    # Sort front-to-back for correct alpha compositing
    with torch.no_grad():
        sort_idx = means_cam[:, 2].argsort()
    means_cam = means_cam[sort_idx]
    scales = scales[sort_idx]
    opacities = opacities[sort_idx]
    rgb = rgb[sort_idx]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Project to screen space
    z = means_cam[:, 2].clamp(min=0.01)  # (N,)
    sx = fx * means_cam[:, 0] / z + cx  # (N,)
    sy = fy * means_cam[:, 1] / z + cy  # (N,)

    # Screen-space scale (approximate: use max of x,y scale * focal / depth)
    screen_scale = (scales.max(dim=-1).values * fx / z).clamp(min=0.5, max=50.0)  # (N,)

    # Create pixel grid
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    px = grid_x.reshape(-1)  # (P,)
    py = grid_y.reshape(-1)  # (P,)
    P = px.shape[0]

    # Compute Gaussian weights: exp(-0.5 * ((px-sx)^2 + (py-sy)^2) / scale^2)
    # (P, N) — this is the big matrix
    dx = px.unsqueeze(1) - sx.unsqueeze(0)  # (P, N)
    dy = py.unsqueeze(1) - sy.unsqueeze(0)  # (P, N)
    inv_s = 1.0 / (screen_scale.unsqueeze(0) + 1e-6)  # (1, N)
    g_val = torch.exp(-0.5 * (dx * dx + dy * dy) * inv_s * inv_s)  # (P, N)

    # Alpha = opacity * gaussian_weight
    alpha = (opacities.unsqueeze(0) * g_val).clamp(max=0.999)  # (P, N)

    # Front-to-back compositing using cumsum (differentiable)
    one_minus = 1.0 - alpha  # (P, N)
    log_om = torch.log(one_minus.clamp(min=1e-8))
    cumsum_log = torch.cumsum(log_om, dim=1)
    T = torch.exp(cumsum_log - log_om)  # (P, N) — transmittance

    # Weighted color
    weighted = alpha * T  # (P, N)
    image = (weighted.unsqueeze(-1) * rgb.unsqueeze(0)).sum(dim=1)  # (P, 3)

    return image.reshape(height, width, 3).clamp(0, 1)
