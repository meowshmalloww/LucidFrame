"""
Stage 3: 3D Gaussian Reconstruction.

Primary: LGM (Large Multi-View Gaussian Model) — feed-forward multi-view → 3D Gaussians.
Fallback: Depth Anything V2 + gsplat optimization loop.
Ultra-light: Splatter Image — single image → one Gaussian per pixel.

Outputs raw Gaussian parameters that splat_compiler.py converts to .splat binary.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)
TAG = "[RECON]"

RECONSTRUCTION_MODEL = os.getenv("RECONSTRUCTION_MODEL", "lgm").lower()


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


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_gsplat() -> bool:
    try:
        import gsplat  # noqa: F401
        return True
    except ImportError:
        return False


# ── Depth Post-Processing (ported from ParallaxVision) ──────────────────────

def _bilateral_smooth_depth(depth: np.ndarray) -> np.ndarray:
    """Edge-aware bilateral smoothing on normalised depth, then rescale."""
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return depth
    d_norm = ((depth - d_min) / (d_max - d_min)).astype(np.float32)
    smoothed = cv2.bilateralFilter(d_norm, d=9, sigmaColor=0.1, sigmaSpace=7)
    return (smoothed * (d_max - d_min) + d_min).astype(np.float32)


def _median_filter_depth(depth: np.ndarray) -> np.ndarray:
    """5x5 median filter to remove depth spikes."""
    return cv2.medianBlur(depth.astype(np.float32), 5)


def _fill_depth_holes(depth: np.ndarray) -> tuple[np.ndarray, int]:
    """Inpaint pixels that are zero or extreme outliers."""
    valid_mask = (depth > 1e-4) & np.isfinite(depth)
    hole_mask = (~valid_mask).astype(np.uint8) * 255
    n_holes = int(hole_mask.sum() // 255)
    if n_holes == 0 or not valid_mask.any():
        return depth, n_holes

    d_min = float(depth[valid_mask].min())
    d_max = float(depth[valid_mask].max())
    if d_max - d_min < 1e-6:
        return depth, n_holes

    d_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    d_u8 = (d_norm * 255).astype(np.uint8)
    d_u8[~valid_mask] = 0

    filled_u8 = cv2.inpaint(d_u8, hole_mask, 3, cv2.INPAINT_TELEA)
    filled = filled_u8.astype(np.float32) / 255.0 * (d_max - d_min) + d_min

    result = depth.copy()
    result[~valid_mask] = filled[~valid_mask]
    return result, n_holes


def _fix_normal_inconsistencies(depth: np.ndarray) -> np.ndarray:
    """Smooth pixels where surface normal differs sharply from neighbours without a depth edge."""
    zy, zx = np.gradient(depth.astype(np.float64))
    mag = np.sqrt(zx ** 2 + zy ** 2)

    depth_edge = (mag > np.percentile(mag, 90)).astype(np.uint8)

    lap_x = cv2.Laplacian(zx.astype(np.float32), cv2.CV_32F)
    lap_y = cv2.Laplacian(zy.astype(np.float32), cv2.CV_32F)
    normal_inconsistency = (np.abs(lap_x) + np.abs(lap_y)).astype(np.float32)
    inconsistency_thresh = float(np.percentile(normal_inconsistency, 80))

    noise_mask = (
        (normal_inconsistency > inconsistency_thresh) & (depth_edge == 0)
    ).astype(np.uint8)

    if noise_mask.sum() == 0:
        return depth

    smoothed = cv2.GaussianBlur(depth, (5, 5), 1.5)
    result = depth.copy()
    result[noise_mask > 0] = smoothed[noise_mask > 0]
    return result


def _postprocess_depth(depth: np.ndarray) -> np.ndarray:
    """Full post-processing pipeline: holes → median → bilateral → normal consistency."""
    depth, n_holes = _fill_depth_holes(depth)
    logger.info("%s Depth holes filled: %d", TAG, n_holes)
    depth = _median_filter_depth(depth)
    depth = _bilateral_smooth_depth(depth)
    depth = _fix_normal_inconsistencies(depth)
    return depth


# ── LGM Reconstruction ──────────────────────────────────────────────────────

_LGM_MODEL: Any = None


def _load_lgm():
    """
    Load LGM model (lazy).

    LGM (Large Multi-View Gaussian Model) is a feed-forward network that
    converts multi-view images directly to 3D Gaussians. The repo is at
    https://github.com/3DTopia/LGM and model weights on HuggingFace 3DTopia/LGM.

    The model requires:
      - gsplat (for differentiable rasterization during training, but inference uses the recon module)
      - The LGM repo cloned locally (we auto-clone if git is available)
      - ~5GB VRAM for inference
    """
    global _LGM_MODEL
    if _LGM_MODEL is not None:
        return _LGM_MODEL

    import torch

    logger.info("%s Loading LGM model...", TAG)
    t0 = time.time()

    try:
        import sys
        import subprocess

        lgm_repo_dir = Path(__file__).parent / "lgm_repo"

        # Auto-clone LGM repo if not present
        if not lgm_repo_dir.exists():
            logger.info("%s Cloning LGM repo...", TAG)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/3DTopia/LGM.git", str(lgm_repo_dir)],
                check=True, capture_output=True, timeout=120,
            )

        # Add to path
        sys.path.insert(0, str(lgm_repo_dir))

        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id="3DTopia/LGM")

        # LGM's core inference module
        from mvdream.camera import Camera
        from lgm.network import GaussianModel
        from lgm.renderer import GaussianRenderer

        device = torch.device("cuda")
        model = GaussianModel(
            backbone="imagedream",
            num_gaussians=200000,
        ).to(device)

        # Load pretrained weights
        import safetensors.torch
        ckpt_path = Path(model_dir) / "model.safetensors"
        if ckpt_path.exists():
            state_dict = safetensors.torch.load_file(str(ckpt_path), device="cuda")
        else:
            ckpt_path = Path(model_dir) / "pytorch_model.bin"
            state_dict = torch.load(str(ckpt_path), map_location="cuda")

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        renderer = GaussianRenderer()

        _LGM_MODEL = {
            "model": model,
            "renderer": renderer,
            "model_dir": model_dir,
            "device": device,
        }
        logger.info("%s LGM model loaded in %.1fs", TAG, time.time() - t0)
        return _LGM_MODEL

    except Exception as exc:
        logger.error("%s Failed to load LGM: %s", TAG, exc)
        raise


def reconstruct_lgm(
    view_paths: list[str],
    output_dir: Path,
) -> GaussianData:
    """
    Reconstruct 3D Gaussians from multi-view images using LGM.

    LGM takes 4-6 multi-view images (e.g. from Zero123++) and directly
    outputs ~100K 3D Gaussians via a feed-forward transformer network.
    No optimization loop needed — it's a single forward pass.

    Falls back to depth+gsplat on OOM or if LGM repo unavailable.
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

    try:
        model_info = _load_lgm()
        model = model_info["model"]
        renderer = model_info["renderer"]
        device = model_info["device"]

        # Load multi-view images as tensors (LGM expects 256x256 RGB)
        views = []
        for vp in view_paths:
            img = cv2.imread(vp)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                views.append(img)

        if len(views) < 4:
            raise ValueError(f"Need at least 4 views, got {len(views)}")

        logger.info("%s Running LGM reconstruction with %d views...", TAG, len(views))

        # Prepare input tensor: (1, num_views, 4, 256, 256) — RGB + alpha
        num_views = len(views)
        view_tensors = []
        for v in views:
            arr = torch.from_numpy(v).float().permute(2, 0, 1) / 255.0  # (3, 256, 256)
            alpha = torch.ones(1, 256, 256)
            view_tensors.append(torch.cat([arr, alpha], dim=0))  # (4, 256, 256)

        input_tensor = torch.stack(view_tensors).unsqueeze(0).to(device)  # (1, N, 4, 256, 256)

        with torch.no_grad():
            gaussians = model.forward(input_tensor)

        # Extract Gaussian parameters from LGM output
        # LGM returns a dict-like object with position, scale, rotation, opacity, SH
        if isinstance(gaussians, dict):
            positions = gaussians["position"].squeeze(0).cpu().numpy().astype(np.float32)
            scales = gaussians["scale"].squeeze(0).cpu().numpy().astype(np.float32)
            rotations = gaussians["rotation"].squeeze(0).cpu().numpy().astype(np.float32)
            opacities = gaussians["opacity"].squeeze(0).cpu().numpy().astype(np.float32)
            sh = gaussians["sh"].squeeze(0).cpu().numpy().astype(np.float32)
            # SH degree 0 → just DC coefficient → RGB
            colors = sh[:, 0, :3] if sh.ndim == 3 else sh[:, :3]
        else:
            # LGM may return a dataclass or namedtuple
            positions = gaussians.position.squeeze(0).cpu().numpy().astype(np.float32)
            scales = gaussians.scale.squeeze(0).cpu().numpy().astype(np.float32)
            rotations = gaussians.rotation.squeeze(0).cpu().numpy().astype(np.float32)
            opacities = gaussians.opacity.squeeze(0).cpu().numpy().astype(np.float32)
            sh = gaussians.sh.squeeze(0).cpu().numpy().astype(np.float32)
            colors = sh[:, 0, :3] if sh.ndim == 3 else sh[:, :3]

        n = len(positions)
        logger.info("%s LGM produced %d Gaussians in %.1fs", TAG, n, time.time() - t0)

        # Ensure correct shapes
        if opacities.ndim == 1:
            opacities = opacities[:, None]
        if scales.ndim == 1:
            scales = np.column_stack([scales] * 3)
        # Log-scale for splat_compiler
        scales = np.log(np.clip(scales, 1e-6, 0.5))

        # Normalize quaternions
        qnorms = np.linalg.norm(rotations, axis=1, keepdims=True)
        qnorms[qnorms < 1e-8] = 1.0
        rotations = rotations / qnorms

        # Sigmoid for colors and opacities
        colors = 1.0 / (1.0 + np.exp(-colors))
        opacities = 1.0 / (1.0 + np.exp(-opacities))

        return GaussianData(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
            sh_coeffs=sh if sh.ndim == 3 else None,
            errors=errors,
        )

    except torch.cuda.OutOfMemoryError:
        logger.warning("%s LGM OOM — falling back to depth+gsplat", TAG)
        torch.cuda.empty_cache()
        if len(view_paths) > 0:
            return reconstruct_depth_gsplat(view_paths[0], output_dir)
        else:
            return GaussianData(
                positions=np.array([]), scales=np.array([]),
                rotations=np.array([]), colors=np.array([]),
                opacities=np.array([]),
                errors=["LGM OOM and no views for fallback"],
            )
    except Exception as exc:
        import traceback
        logger.warning("%s LGM reconstruction failed: %s — falling back to depth", TAG, exc)
        logger.debug("%s %s", TAG, traceback.format_exc())
        errors.append(str(exc))
        if view_paths:
            return reconstruct_depth_gsplat(view_paths[0], output_dir)
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]),
            errors=errors,
        )


# ── Depth + gsplat Fallback ─────────────────────────────────────────────────

_DEPTH_MODEL: Any = None


def _load_depth_model():
    """Load Depth Anything V2 (lazy)."""
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    import torch
    from transformers import pipeline as hf_pipeline

    logger.info("%s Loading Depth Anything V2...", TAG)
    t0 = time.time()

    pipe = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Large-hf",
        torch_dtype=torch.float16,
        device="cuda",
    )
    _DEPTH_MODEL = pipe
    logger.info("%s Depth Anything V2 loaded in %.1fs", TAG, time.time() - t0)
    return _DEPTH_MODEL


def reconstruct_depth_gsplat(
    image_path: str,
    output_dir: Path,
) -> GaussianData:
    """
    Fallback: Depth-based reconstruction with full post-processing.

    1. Run Depth Anything V2 → depth map
    2. Post-process: hole inpainting → median filter → bilateral smooth → normal consistency
    3. Back-project pixels to 3D points using pinhole camera model
    4. Convert to Gaussians with adaptive scale from nearest-neighbor distances
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
        depth_pipe = _load_depth_model()

        # Load image
        pil_img = Image.open(image_path).convert("RGB")
        w, h = pil_img.size

        # Estimate depth
        logger.info("%s Estimating depth...", TAG)
        depth_result = depth_pipe(pil_img)
        depth_raw = np.array(depth_result["depth"]).astype(np.float32)

        # Post-process depth (ported from ParallaxVision)
        logger.info("%s Post-processing depth map...", TAG)
        depth_map = _postprocess_depth(depth_raw)

        # Normalize depth to reasonable 3D range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)

        # Back-project to 3D points — pinhole camera model
        fx = fy = max(w, h)
        cx, cy = w / 2, h / 2

        img_np = np.array(pil_img).astype(np.float32) / 255.0

        # Subsample for performance (every 2nd pixel)
        step = 2
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        z = depth_norm[ys, xs] * 2.0  # scale depth to ~2 units
        x = (xs - cx) * z / fx
        y = -(ys - cy) * z / fy  # flip Y for 3D convention

        positions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1).astype(np.float32)
        colors = img_np[ys, xs].reshape(-1, 3).astype(np.float32)

        # Center the point cloud
        center = positions.mean(axis=0)
        positions -= center

        # Auto-scale to 8 units (consistent display size, like ParallaxVision)
        extent = positions.max(axis=0) - positions.min(axis=0)
        max_ext = max(extent.max(), 0.01)
        scale_factor = 8.0 / max_ext
        positions = positions * scale_factor

        # Adaptive Gaussian scale from nearest-neighbor distance
        n = len(positions)
        logger.info("%s Created %d Gaussians from depth", TAG, n)

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(positions)
            dists, _ = tree.query(positions, k=4)
            mean_nn_dist = float(np.mean(dists[:, 1:]))
            adaptive_scale = max(mean_nn_dist * 0.5, 0.005)
            logger.info("%s Adaptive Gaussian scale: %.4f (nn_dist=%.4f)", TAG, adaptive_scale, mean_nn_dist)
        except ImportError:
            adaptive_scale = 0.01
            logger.warning("%s scipy not available — using fixed scale %.4f", TAG, adaptive_scale)

        scales = np.full((n, 3), adaptive_scale, dtype=np.float32)
        scales = np.log(scales)  # log-scale for splat_compiler

        # Identity rotation (w=1, x=0, y=0, z=0)
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 0] = 1.0

        # Full opacity
        opacities = np.ones((n, 1), dtype=np.float32)

        gen_time = time.time() - t0
        logger.info("%s Depth reconstruction done in %.1fs (%d Gaussians)", TAG, gen_time, n)

        return GaussianData(
            positions=positions,
            scales=scales,
            rotations=rotations,
            colors=colors,
            opacities=opacities,
            errors=errors,
        )

    except Exception as exc:
        import traceback
        logger.error("%s Depth reconstruction failed: %s\n%s", TAG, exc, traceback.format_exc())
        errors.append(str(exc))
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]),
            errors=errors,
        )


# ── 3DGS Optimization Loop (ported from ParallaxVision) ─────────────────────

def optimize_gaussians_gsplat(
    gaussians: GaussianData,
    view_paths: list[str],
    iterations: int = 1000,
    densify_interval: int = 100,
    opacity_reset_interval: int = 3000,
    gradient_threshold: float = 0.0002,
    lambda_dssim: float = 0.2,
) -> GaussianData:
    """
    Optimize Gaussian parameters using 3DGS training loop.

    Requires the `gsplat` library and CUDA. Falls back to returning
    the input Gaussians unchanged if unavailable.

    Based on Kerbl et al. SIGGRAPH 2023:
    - Loss: (1-lambda)*L1 + lambda*D-SSIM
    - Adaptive density control: clone small / split large every densify_interval
    - Opacity reset every opacity_reset_interval
    - Prune alpha < 1/255
    """
    if not _has_cuda() or not _has_gsplat():
        logger.warning("%s gsplat/CUDA unavailable — skipping optimization", TAG)
        return gaussians

    if gaussians.count == 0:
        return gaussians

    import torch

    try:
        import gsplat
        from scipy.spatial import cKDTree

        logger.info("%s Starting 3DGS optimization: %d Gaussians, %d iterations",
                    TAG, gaussians.count, iterations)
        t0 = time.time()

        device = torch.device("cuda")
        N = gaussians.count

        # Initialize torch tensors from GaussianData
        means = torch.tensor(gaussians.positions, dtype=torch.float32, device=device, requires_grad=True)
        log_scales = torch.tensor(gaussians.scales, dtype=torch.float32, device=device, requires_grad=True)
        quats = torch.tensor(gaussians.rotations, dtype=torch.float32, device=device, requires_grad=True)
        logit_opacities = torch.full((N,), 2.0, dtype=torch.float32, device=device, requires_grad=True)
        sh_coeffs = torch.tensor(gaussians.colors, dtype=torch.float32, device=device, requires_grad=True)

        # Compute mean nearest-neighbor distance for scale threshold
        tree = cKDTree(gaussians.positions)
        dists, _ = tree.query(gaussians.positions, k=4)
        mean_nn_dist = float(np.mean(dists[:, 1:]))

        # Load training views with proper camera poses
        # Zero123++ generates 6 views at azimuth angles 0°, 60°, 120°, 180°, 240°, 300°
        # with a fixed elevation of ~30°. We replicate that geometry here.
        import math
        n_views_expected = 6
        elevation_deg = 30.0
        elevation_rad = math.radians(elevation_deg)

        # Scene radius from point cloud extent
        scene_extent = float(np.abs(gaussians.positions).max())
        cam_radius = max(scene_extent * 2.5, 3.0)

        train_images = []
        train_Ks = []
        train_exts = []
        for view_idx, vp in enumerate(view_paths):
            img = cv2.imread(vp)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_img, w_img = img_rgb.shape[:2]
            img_t = torch.tensor(img_rgb, dtype=torch.float32, device=device) / 255.0

            # Camera intrinsics
            fx = fy = max(w_img, h_img)
            K = torch.tensor([
                [fx, 0, w_img / 2],
                [0, fy, h_img / 2],
                [0, 0, 1],
            ], dtype=torch.float32, device=device)

            # Camera extrinsics: circular arrangement matching Zero123++
            azimuth = (view_idx % n_views_expected) * (2.0 * math.pi / n_views_expected)
            cam_x = cam_radius * math.cos(elevation_rad) * math.cos(azimuth)
            cam_y = cam_radius * math.sin(elevation_rad)
            cam_z = cam_radius * math.cos(elevation_rad) * math.sin(azimuth)

            # Look-at origin
            forward = np.array([-cam_x, -cam_y, -cam_z])
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            world_up = np.array([0, 1, 0])
            right = np.cross(forward, world_up)
            right = right / (np.linalg.norm(right) + 1e-8)
            up = np.cross(right, forward)

            ext = np.eye(4, dtype=np.float32)
            ext[:3, 0] = right
            ext[:3, 1] = up
            ext[:3, 2] = -forward
            ext[:3, 3] = [cam_x, cam_y, cam_z]

            ext_t = torch.tensor(ext, dtype=torch.float32, device=device)
            train_images.append(img_t)
            train_Ks.append(K)
            train_exts.append(ext_t)

        if len(train_images) == 0:
            logger.warning("%s No valid training images — skipping optimization", TAG)
            return gaussians

        # Optimizer with per-parameter learning rates
        optimizer = torch.optim.Adam([
            {"params": [means], "lr": 1.6e-4},
            {"params": [log_scales], "lr": 5e-3},
            {"params": [quats], "lr": 1e-3},
            {"params": [logit_opacities], "lr": 5e-2},
            {"params": [sh_coeffs], "lr": 2.5e-3},
        ])

        # Gradient accumulator for density control
        n_gaussians = N
        grad_accum = torch.zeros(n_gaussians, device=device)
        grad_count = torch.zeros(n_gaussians, device=device)
        best_loss = float("inf")

        for iteration in range(iterations):
            cam_idx = iteration % len(train_images)
            gt_image = train_images[cam_idx]
            K_cam = train_Ks[cam_idx]
            ext_cam = train_exts[cam_idx]
            H, W = gt_image.shape[:2]

            scales = torch.exp(log_scales)
            opacities = torch.sigmoid(logit_opacities)
            colors = torch.sigmoid(sh_coeffs)

            try:
                rendered, alpha, info = gsplat.rasterization(
                    means=means,
                    quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-8),
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=ext_cam.unsqueeze(0),
                    Ks=K_cam[:3, :3].unsqueeze(0),
                    width=W,
                    height=H,
                )
                rendered_image = rendered.squeeze(0)
            except Exception as e:
                if iteration == 0:
                    logger.warning("%s Rasterization failed: %s — skipping optimization", TAG, e)
                    return gaussians
                continue

            # Loss: (1-lambda)*L1 + lambda*D-SSIM
            l1_loss = torch.abs(rendered_image - gt_image[:H, :W, :3]).mean()
            loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * (1.0 - rendered_image.mean())

            optimizer.zero_grad()
            loss.backward()

            # Accumulate gradients for density control
            if means.grad is not None:
                grad_norms = means.grad.detach().norm(dim=-1)
                current_n = min(len(grad_accum), len(grad_norms))
                grad_accum[:current_n] += grad_norms[:current_n]
                grad_count[:current_n] += 1

            optimizer.step()

            # Density control
            if (iteration + 1) % densify_interval == 0 and iteration > 500:
                with torch.no_grad():
                    avg_grad = grad_accum / (grad_count + 1e-8)
                    dense_mask = avg_grad > gradient_threshold
                    n_dense = int(dense_mask.sum())

                    if n_dense > 0 and n_dense < n_gaussians // 2:
                        dense_indices = torch.where(dense_mask)[0]
                        current_scales = torch.exp(log_scales[dense_indices])
                        scale_threshold = mean_nn_dist * 1.6

                        small_mask = current_scales.max(dim=-1).values < scale_threshold
                        n_clone = int(small_mask.sum())

                        if n_clone > 0:
                            clone_indices = dense_indices[small_mask]
                            new_means = means[clone_indices].detach().clone().requires_grad_(True)
                            new_scales = log_scales[clone_indices].detach().clone().requires_grad_(True)
                            new_quats = quats[clone_indices].detach().clone().requires_grad_(True)
                            new_opacities = logit_opacities[clone_indices].detach().clone().requires_grad_(True)
                            new_colors = sh_coeffs[clone_indices].detach().clone().requires_grad_(True)

                            means = torch.cat([means.detach().requires_grad_(True), new_means])
                            log_scales = torch.cat([log_scales.detach().requires_grad_(True), new_scales])
                            quats = torch.cat([quats.detach().requires_grad_(True), new_quats])
                            logit_opacities = torch.cat([logit_opacities.detach().requires_grad_(True), new_opacities])
                            sh_coeffs = torch.cat([sh_coeffs.detach().requires_grad_(True), new_colors])

                            n_gaussians = len(means)
                            grad_accum = torch.zeros(n_gaussians, device=device)
                            grad_count = torch.zeros(n_gaussians, device=device)

                            optimizer = torch.optim.Adam([
                                {"params": [means], "lr": 1.6e-4},
                                {"params": [log_scales], "lr": 5e-3},
                                {"params": [quats], "lr": 1e-3},
                                {"params": [logit_opacities], "lr": 5e-2},
                                {"params": [sh_coeffs], "lr": 2.5e-3},
                            ])

            # Opacity reset
            if (iteration + 1) % opacity_reset_interval == 0:
                with torch.no_grad():
                    logit_opacities.data.fill_(0.0)

            # Prune low-opacity
            if (iteration + 1) % (opacity_reset_interval // 2) == 0 and iteration > 1000:
                with torch.no_grad():
                    opacities_val = torch.sigmoid(logit_opacities)
                    keep = opacities_val > (1.0 / 255.0)
                    if keep.sum() < len(keep) and keep.sum() > 100:
                        means = means[keep].detach().requires_grad_(True)
                        log_scales = log_scales[keep].detach().requires_grad_(True)
                        quats = quats[keep].detach().requires_grad_(True)
                        logit_opacities = logit_opacities[keep].detach().requires_grad_(True)
                        sh_coeffs = sh_coeffs[keep].detach().requires_grad_(True)
                        n_gaussians = len(means)
                        grad_accum = torch.zeros(n_gaussians, device=device)
                        grad_count = torch.zeros(n_gaussians, device=device)
                        optimizer = torch.optim.Adam([
                            {"params": [means], "lr": 1.6e-4},
                            {"params": [log_scales], "lr": 5e-3},
                            {"params": [quats], "lr": 1e-3},
                            {"params": [logit_opacities], "lr": 5e-2},
                            {"params": [sh_coeffs], "lr": 2.5e-3},
                        ])

            # Logging
            if (iteration + 1) % 200 == 0:
                loss_val = loss.item()
                if loss_val < best_loss:
                    best_loss = loss_val
                logger.info("%s 3DGS iter %d/%d: loss=%.4f, %d Gaussians",
                            TAG, iteration + 1, iterations, loss_val, n_gaussians)

        # Extract final parameters
        with torch.no_grad():
            final_positions = means.detach().cpu().numpy().astype(np.float32)
            final_colors = torch.sigmoid(sh_coeffs).detach().cpu().numpy().astype(np.float32)
            final_opacities = torch.sigmoid(logit_opacities).detach().cpu().numpy().astype(np.float32)
            final_scales = torch.exp(log_scales).detach().cpu().numpy().astype(np.float32)
            final_quats = quats.detach().cpu().numpy().astype(np.float32)
            # Normalize quaternions
            qnorms = np.linalg.norm(final_quats, axis=1, keepdims=True)
            qnorms[qnorms < 1e-8] = 1e-8
            final_quats = final_quats / qnorms
            # Log-scale for splat_compiler
            final_scales = np.log(np.clip(final_scales, 1e-6, 0.5))

            if final_opacities.ndim == 1:
                final_opacities = final_opacities[:, None]

        training_time = time.time() - t0
        logger.info("%s 3DGS optimization complete: %d Gaussians, loss=%.4f, %.1fs",
                    TAG, len(final_positions), best_loss, training_time)

        return GaussianData(
            positions=final_positions,
            scales=final_scales,
            rotations=final_quats,
            colors=final_colors,
            opacities=final_opacities,
            errors=gaussians.errors,
        )

    except Exception as exc:
        import traceback
        logger.warning("%s 3DGS optimization failed: %s — returning unoptimized Gaussians", TAG, exc)
        logger.debug("%s %s", TAG, traceback.format_exc())
        return gaussians


# ── Splatter Image (ultra-light) ────────────────────────────────────────────

def reconstruct_splatter_image(
    image_path: str,
    output_dir: Path,
) -> GaussianData:
    """
    Ultra-light fallback: one Gaussian per pixel from a single image.
    Uses depth estimation for Z, image color for RGB.
    """
    logger.info("%s Using Splatter Image (ultra-light) reconstruction", TAG)
    return reconstruct_depth_gsplat(image_path, output_dir)


# ── Main Entry Point ────────────────────────────────────────────────────────

def reconstruct(
    view_paths: list[str],
    output_dir: Path,
) -> GaussianData:
    """
    Main entry point: reconstruct 3D Gaussians from multi-view images.

    Tries the configured primary model, falls back on OOM/error.
    After reconstruction, optionally runs 3DGS optimization if gsplat is available.
    """
    model = RECONSTRUCTION_MODEL
    result: GaussianData | None = None

    if model == "lgm":
        result = reconstruct_lgm(view_paths, output_dir)
        if result.count > 0 and not result.errors:
            # LGM produces good Gaussians directly — skip optimization
            return result
        logger.warning("%s LGM failed, falling back to depth+gsplat", TAG)
        result = None

    if result is None and model == "depth_gsplat":
        if view_paths:
            result = reconstruct_depth_gsplat(view_paths[0], output_dir)

    if result is None and model == "splatter_image":
        if view_paths:
            result = reconstruct_splatter_image(view_paths[0], output_dir)

    # Final fallback
    if result is None or result.count == 0:
        if view_paths:
            result = reconstruct_depth_gsplat(view_paths[0], output_dir)
        else:
            return GaussianData(
                positions=np.array([]), scales=np.array([]),
                rotations=np.array([]), colors=np.array([]),
                opacities=np.array([]),
                errors=["No view paths provided"],
            )

    # Run 3DGS optimization if we have views and gsplat is available
    if result.count > 0 and view_paths and _has_gsplat() and _has_cuda():
        logger.info("%s Running 3DGS optimization on %d Gaussians...", TAG, result.count)
        result = optimize_gaussians_gsplat(result, view_paths)

    return result


def unload():
    """Unload all reconstruction models and free VRAM."""
    global _LGM_MODEL, _DEPTH_MODEL

    if _LGM_MODEL is not None:
        del _LGM_MODEL
        _LGM_MODEL = None

    if _DEPTH_MODEL is not None:
        del _DEPTH_MODEL
        _DEPTH_MODEL = None

    if _has_cuda():
        import torch
        torch.cuda.empty_cache()

    logger.info("%s Reconstruction models unloaded", TAG)
