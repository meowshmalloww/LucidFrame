"""
Flash3D inference wrapper: single image → multi-layer 3D Gaussians → .splat file.

Flash3D (3DV 2025, Oxford) predicts K layers of 3D Gaussians per pixel from a
single image, where layer 1 follows the depth map and layers 2+ model occluded
and unseen regions. This is a feed-forward model — no optimization needed.

Usage:
    from flash3d_wrapper import reconstruct_flash3d
    gaussians = reconstruct_flash3d("path/to/image.png")
"""
import sys
import os
import math
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

logger = logging.getLogger(__name__)
TAG = "[Flash3D]"

# Path to the Flash3D repo
FLASH3D_DIR = Path(__file__).parent.parent / "external" / "flash3d"
MODEL_CKPT = FLASH3D_DIR / "exp" / "re10k_v2" / "checkpoints" / "model_re10k_v2.pth"


def _ensure_flash3d_importable():
    """Add Flash3D to sys.path so we can import its modules."""
    repo = str(FLASH3D_DIR)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    logger.info("%s Flash3D repo path: %s", TAG, repo)


def _build_cfg():
    """Build the OmegaConf config that Flash3D's model expects."""
    from omegaconf import OmegaConf

    # Start with the base config, then apply the layered_re10k experiment overrides
    cfg = OmegaConf.load(str(FLASH3D_DIR / "configs" / "config.yaml"))

    # Load model config
    model_cfg = OmegaConf.load(str(FLASH3D_DIR / "configs" / "model" / "gaussian.yaml"))
    backbone_cfg = OmegaConf.load(str(FLASH3D_DIR / "configs" / "model" / "backbone" / "resnet.yaml"))
    depth_cfg = OmegaConf.load(str(FLASH3D_DIR / "configs" / "model" / "depth" / "unidepth.yaml"))

    # Merge model configs
    cfg.model = OmegaConf.merge(model_cfg, OmegaConf.create({
        "backbone": backbone_cfg,
        "depth": depth_cfg,
    }))

    # Apply experiment overrides (layered_re10k)
    exp_cfg = OmegaConf.load(str(FLASH3D_DIR / "configs" / "experiment" / "layered_re10k.yaml"))
    # The experiment config has model overrides under 'model:' key
    if "model" in exp_cfg:
        for k, v in exp_cfg.model.items():
            cfg.model[k] = v

    # Dataset settings for inference
    cfg.dataset = OmegaConf.create({
        "name": "re10k",
        "width": 384,
        "height": 256,
        "pad_border_aug": 32,
        "scale_pose_by_depth": True,
    })

    # Disable gaussian_rendering — we only need Gaussian parameters, not rendered images
    cfg.model.gaussian_rendering = False

    # Set data_loader batch_size to 1 for inference
    cfg.data_loader = OmegaConf.create({"batch_size": 1, "num_workers": 0})

    return cfg


def _patch_nystrom_attention():
    """Patch UniDepth's NystromAttention to use standard SDPA instead of xformers.
    This must be done AFTER torch.hub.load downloads the repo but BEFORE the
    UniDepth model is instantiated. We do it by patching the module in-place.
    """
    import importlib
    import torch.nn as nn
    import torch.nn.functional as F

    class StandardAttention(nn.Module):
        """Drop-in replacement for NystromAttention using standard SDPA."""
        def __init__(self, num_landmarks=128, num_heads=4, dropout=0.0, **kwargs):
            super().__init__()
            self.num_heads = num_heads
            self.dropout = dropout

        def __call__(self, q, k, v, key_padding_mask=None, **kwargs):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if key_padding_mask is not None:
                attn_mask = key_padding_mask[:, None, None, :].expand(-1, self.num_heads, -1, -1)
                attn_mask = attn_mask.to(q.dtype) * -1e4
            else:
                attn_mask = None
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0
            )
            return out.transpose(1, 2)

    # Try to patch the already-loaded module
    try:
        mod = importlib.import_module("unidepth.layers.nystrom_attention")
        mod.NystromAttention = StandardAttention
        logger.info("%s Patched NystromAttention with standard SDPA", TAG)
    except ImportError:
        pass

    # Also patch the path in sys.modules that might be used
    for key, mod in list(sys.modules.items()):
        if "nystrom_attention" in key and hasattr(mod, 'NystromAttention'):
            mod.NystromAttention = StandardAttention
            logger.info("%s Patched NystromAttention in %s", TAG, key)


def _load_model(cfg, device="cuda"):
    """Load the pretrained Flash3D model."""
    from models.model import GaussianPredictor

    # Pre-load UniDepth to trigger torch.hub download, then patch NystromAttention
    # The UniDepth model is loaded inside GaussianPredictor.__init__
    # We need to patch after the hub repo is downloaded but before NystromBlock is created

    model = GaussianPredictor(cfg).to(device)

    if MODEL_CKPT.exists():
        logger.info("%s Loading pretrained model from %s", TAG, MODEL_CKPT)
        ckpt = torch.load(str(MODEL_CKPT), map_location=device, weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Handle EMA model
        if any(k.startswith("ema_model.") for k in state_dict):
            # Use EMA weights for inference
            ema_state = {}
            for k, v in state_dict.items():
                if k.startswith("ema_model."):
                    ema_state[k.replace("ema_model.", "")] = v
            if ema_state:
                state_dict = ema_state

        # Filter out backproject_depth buffers — they are precomputed pixel
        # coordinates that depend on batch_size, not learned weights.
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith("backproject_depth.")}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        expected_missing = [
            key for key in missing
            if key.startswith("models.unidepth_extended.unidepth.")
        ]
        unexpected_missing = [key for key in missing if key not in expected_missing]
        if expected_missing:
            logger.info("%s UniDepth weights loaded separately: %d checkpoint keys absent", TAG, len(expected_missing))
        if unexpected_missing:
            logger.warning("%s Missing Flash3D keys: %d", TAG, len(unexpected_missing))
        if unexpected:
            logger.warning("%s Unexpected keys: %d", TAG, len(unexpected))
    else:
        logger.error("%s Model checkpoint not found at %s", TAG, MODEL_CKPT)
        raise FileNotFoundError(f"Flash3D checkpoint not found: {MODEL_CKPT}")

    model.eval()
    return model


def _prepare_image(image_path: str, target_w=384, target_h=256, pad_border=32):
    """Load and preprocess image for Flash3D.

    Flash3D expects images at a specific resolution (multiples of 32) with
    border padding. Returns the padded image tensor and original PIL image.
    """
    pil_img = Image.open(image_path).convert("RGB")

    # Resize to target resolution while maintaining aspect ratio
    # Flash3D uses 384x256 (3:2 aspect ratio)
    img_w, img_h = pil_img.size
    # Choose the best fit - use 384x256 but allow other multiples of 32
    # For better quality, use larger resolution if GPU allows
    target_w = 384
    target_h = 256

    # Resize preserving aspect ratio, then crop/pad to exact target
    pil_resized = pil_img.resize((target_w, target_h), Image.LANCZOS)

    # Convert to tensor [1, 3, H, W] in [0, 1]
    img_np = np.array(pil_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

    # Pad with border (Flash3D uses pad_border_aug=32)
    if pad_border > 0:
        img_tensor = F.pad(img_tensor, [pad_border, pad_border, pad_border, pad_border], mode="reflect")

    return img_tensor, pil_resized


def reconstruct_flash3d(
    image_path: str,
    output_dir: Optional[Path] = None,
    device: str = "cuda",
) -> "GaussianData":
    """
    Reconstruct 3D Gaussians from a single image using Flash3D.

    Args:
        image_path: Path to input image
        output_dir: Optional directory to save intermediate results
        device: torch device ('cuda' or 'cpu')

    Returns:
        GaussianData with positions, scales, rotations, colors, opacities
    """
    from reconstruction_stage import GaussianData

    t0 = time.time()
    errors = []

    if not _has_cuda():
        errors.append("CUDA not available — Flash3D requires CUDA")
        return GaussianData(
            positions=np.array([]), scales=np.array([]),
            rotations=np.array([]), colors=np.array([]),
            opacities=np.array([]), errors=errors,
        )

    _ensure_flash3d_importable()

    # Download model if not present
    if not MODEL_CKPT.exists():
        logger.info("%s Downloading pretrained model from HuggingFace...", TAG)
        from huggingface_hub import hf_hub_download
        MODEL_CKPT.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="einsafutdinov/flash3d",
            filename="model_re10k_v2.pth",
            local_dir=str(MODEL_CKPT.parent),
        )

    # Build config and load model
    cfg = _build_cfg()
    model = _load_model(cfg, device=device)

    # Prepare image
    pad_border = cfg.dataset.pad_border_aug
    img_tensor, pil_orig = _prepare_image(image_path,
                                           target_w=cfg.dataset.width,
                                           target_h=cfg.dataset.height,
                                           pad_border=pad_border)
    img_tensor = img_tensor.to(device)

    H = cfg.dataset.height
    W = cfg.dataset.width
    H_padded = H + 2 * pad_border
    W_padded = W + 2 * pad_border

    logger.info("%s Input image: %dx%d (padded: %dx%d)", TAG, W, H, W_padded, H_padded)

    # Build inputs dict that Flash3D expects
    # Flash3D uses tuple keys like ("color_aug", 0, 0) not nested dicts
    # Don't pass K_src — let UniDepth estimate intrinsics from the image
    inputs = {}
    inputs[("color_aug", 0, 0)] = img_tensor
    inputs[("color", 0, 0)] = img_tensor
    inputs[("color_orig_res", 0, 0)] = img_tensor
    inputs["target_frame_ids"] = [1]  # We only need the source frame

    # Run the model
    with torch.no_grad():
        outputs = model(inputs)

    # Extract Gaussian parameters
    # outputs contains:
    #   gauss_means: [B*K, 4, N]  — 3D positions (homogeneous)
    #   gauss_opacity: [B*K, 1, H, W] — opacity after sigmoid
    #   gauss_scaling: [B*K, 3, H, W] — scale after exp * lambda
    #   gauss_rotation: [B*K, 4, H, W] — rotation quaternion (normalized)
    #   gauss_features_dc: [B*K, 3, H, W] — DC color (SH band 0)
    #   gauss_offset: [B*K, 3, H, W] — positional offset
    #
    # The model outputs at padded resolution (H_padded x W_padded).
    # We crop to the original image area (H x W) to discard border Gaussians,
    # which are extrapolated from reflect-padded pixels and produce outlier positions.

    gaussians_per_pixel = cfg.model.gaussians_per_pixel  # 2
    active_layers = int(os.getenv("FLASH3D_LAYERS", "2"))
    active_layers = max(1, min(active_layers, gaussians_per_pixel))
    sl_h = slice(pad_border, pad_border + H)  # crop rows: [32:288]
    sl_w = slice(pad_border, pad_border + W)  # crop cols: [32:416]

    # Extract means (positions) — reshape from flattened to spatial for cropping
    means = outputs["gauss_means"]  # [B*K, 4, H_padded*W_padded]
    means = means[:, :3, :]  # [B*K, 3, H_padded*W_padded]
    means = means.reshape(means.shape[0], 3, H_padded, W_padded)
    means = means[:, :, sl_h, sl_w]  # [B*K, 3, H, W]
    means = means.reshape(means.shape[0], 3, H * W)
    means = rearrange(means, "(b k) c n -> b (k n) c", k=gaussians_per_pixel)
    positions = means[0].cpu().numpy()  # [K*H*W, 3]

    # Extract opacity
    opacity = outputs["gauss_opacity"][:, :, sl_h, sl_w]  # [B*K, 1, H, W]
    opacity = rearrange(opacity, "(b k) c h w -> b (k h w) c", k=gaussians_per_pixel)
    opacities = opacity[0, :, 0].cpu().numpy()  # [K*H*W]

    # Extract scale
    scaling = outputs["gauss_scaling"][:, :, sl_h, sl_w]  # [B*K, 3, H, W]
    scaling = rearrange(scaling, "(b k) c h w -> b (k h w) c", k=gaussians_per_pixel)
    scales_linear = scaling[0].cpu().numpy()  # [K*H*W, 3]

    # Extract rotation
    rotation = outputs["gauss_rotation"][:, :, sl_h, sl_w]  # [B*K, 4, H, W]
    rotation = rearrange(rotation, "(b k) c h w -> b (k h w) c", k=gaussians_per_pixel)
    rotations = rotation[0].cpu().numpy()  # [K*H*W, 4]

    # Extract color (from SH DC component)
    feat_dc = outputs["gauss_features_dc"][:, :, sl_h, sl_w]  # [B*K, 3, H, W]
    feat_dc = rearrange(feat_dc, "(b k) c h w -> b (k h w) c", k=gaussians_per_pixel)
    sh_c0 = 0.28209479
    colors = (sh_c0 * feat_dc[0] + 0.5).clamp(0, 1).cpu().numpy()  # [K*H*W, 3]

    # Keep the photographed view visually crisp. Flash3D predicts colors at its
    # fixed 384x256 training grid; blending the first (visible) layer with the
    # Lanczos-resampled source avoids compounding decoder softness without
    # inventing extra geometry. Occluded layers retain their predicted colors.
    try:
        source_anchor = float(os.getenv("FLASH3D_SOURCE_COLOR_ANCHOR", "0.72"))
    except ValueError:
        source_anchor = 0.72
    source_anchor = float(np.clip(source_anchor, 0.0, 1.0))
    if source_anchor > 0.0:
        source_colors = np.asarray(pil_orig, dtype=np.float32).reshape(H * W, 3) / 255.0
        colors[: H * W] = (
            source_anchor * source_colors
            + (1.0 - source_anchor) * colors[: H * W]
        )

    if active_layers < gaussians_per_pixel:
        layer_indices = np.concatenate([
            np.arange(layer * H * W, (layer + 1) * H * W)
            for layer in range(active_layers)
        ])
        positions = positions[layer_indices]
        opacities = opacities[layer_indices]
        scales_linear = scales_linear[layer_indices]
        rotations = rotations[layer_indices]
        colors = colors[layer_indices]

    # ── Post-processing: normalize scene, fix Y-axis, size scales ───────────
    # Flash3D and gsplat 1.2.9 both use X=right, Y=down, Z=forward camera
    # coordinates. A Y reflection here would invert the image in the browser.
    # Scale coverage is calibrated after the scene has been normalized.

    # 1. Center scene at median (same as official export_ply)
    center = np.median(positions, axis=0)
    positions = positions - center

    # 2. Normalize by 95th percentile (same as official export_ply)
    scale_factor = np.quantile(np.abs(positions), 0.95, axis=0).max()
    if scale_factor < 1e-8:
        scale_factor = 1.0
    positions = positions / scale_factor
    scales_linear = scales_linear / scale_factor

    # 2.5. Filter outlier positions — after normalization, 95% of points are in [-1, 1].
    # Remove extreme outliers beyond [-2, 2] which are typically from Flash3D's
    # second layer (occluded regions) with unreliable depth predictions.
    # This is critical: the gsplat viewer places the camera at Z=-3 and expects
    # the scene in [-1, 1]; outliers at Z=4 make the camera appear inside the scene.
    outlier_mask = (np.abs(positions) <= 2.0).all(axis=1)
    n_outliers = len(positions) - int(outlier_mask.sum())
    if n_outliers > 0:
        positions = positions[outlier_mask]
        scales_linear = scales_linear[outlier_mask]
        rotations = rotations[outlier_mask]
        colors = colors[outlier_mask]
        opacities = opacities[outlier_mask]
        logger.info("%s Filtered %d outlier Gaussians (|pos| > 2.0 after normalization)",
                    TAG, n_outliers)

    # 3. Preserve Flash3D camera axes and encode the quaternion for .splat.
    # Correction: keep the shared camera frame; do not reflect coordinates.
    # Flash3D quaternions are [x, y, z, w]; gsplat binary quaternions are
    # stored as [w, x, y, z].
    rotations = rotations[:, [3, 0, 1, 2]]

    # 4. Fit the robust median scale to a little over one source-pixel spacing.
    # Smaller footprints create porous patches during motion. Coverage closes
    # sampling gaps only; it cannot truthfully fill occluded geometry.
    # FLASH3D_SCALE_BOOST overrides calibration; FLASH3D_COVERAGE adjusts it.
    # The default coverage is 1.04 source pixels: enough overlap without excess blur.
    scene_width = float(np.quantile(positions[:, 0], 0.99) - np.quantile(positions[:, 0], 0.01))
    pixel_spacing = max(scene_width / W, 1e-5)
    try:
        coverage = float(os.getenv("FLASH3D_COVERAGE", "1.04"))
    except ValueError:
        coverage = 1.04
    coverage = float(np.clip(coverage, 0.50, 1.75))
    target_scale = pixel_spacing * coverage
    robust_median = max(float(np.median(scales_linear)), 1e-8)
    configured_boost = os.getenv("FLASH3D_SCALE_BOOST")
    scale_boost = (
        max(float(configured_boost), 0.1)
        if configured_boost is not None
        else target_scale / robust_median
    )
    scales_linear = scales_linear * scale_boost

    # 6. Log stats for debugging
    logger.info("%s Scene scale_factor: %.4f, scale boost: %.1fx, coverage: %.2f px",
                TAG, scale_factor, scale_boost, coverage)
    logger.info("%s Positions range: [%.3f, %.3f] [%.3f, %.3f] [%.3f, %.3f]",
                TAG,
                positions[:, 0].min(), positions[:, 0].max(),
                positions[:, 1].min(), positions[:, 1].max(),
                positions[:, 2].min(), positions[:, 2].max())
    logger.info("%s Scales (linear) median: [%.6f, %.6f, %.6f]",
                TAG,
                np.median(scales_linear[:, 0]),
                np.median(scales_linear[:, 1]),
                np.median(scales_linear[:, 2]))

    # 7. Filter non-finite values and prune low-opacity Gaussians
    # (enforce_geometry_quality is not called for Flash3D, so we do basic filtering here)
    finite = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(scales_linear).all(axis=1)
        & np.isfinite(rotations).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(opacities)
    )
    n_before = len(positions)
    if not finite.all():
        positions = positions[finite]
        scales_linear = scales_linear[finite]
        rotations = rotations[finite]
        colors = colors[finite]
        opacities = opacities[finite]
        logger.warning("%s Filtered %d non-finite Gaussians", TAG, n_before - len(positions))

    # Prune near-zero opacity (they contribute nothing but add render cost)
    opacity_mask = opacities >= 0.01
    n_before_op = len(positions)
    if not opacity_mask.all():
        positions = positions[opacity_mask]
        scales_linear = scales_linear[opacity_mask]
        rotations = rotations[opacity_mask]
        colors = colors[opacity_mask]
        opacities = opacities[opacity_mask]
        logger.info("%s Pruned %d low-opacity Gaussians", TAG, n_before_op - len(positions))

    n = len(positions)
    logger.info("%s Flash3D generated %d Gaussians (%d/%d layers, %dx%d image)",
                TAG, n, active_layers, gaussians_per_pixel, W, H)

    # Convert to log-scale for our GaussianData format
    scales_log = np.log(np.clip(scales_linear, 1e-6, 0.05)).astype(np.float32)

    # Normalize quaternions
    rot_norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (rot_norms + 1e-8)

    # Opacities as [N, 1]
    opacities = opacities[:, None].astype(np.float32)

    gen_time = time.time() - t0
    logger.info("%s Flash3D reconstruction done in %.1fs (%d Gaussians)", TAG, gen_time, len(positions))

    result = GaussianData(
        positions=positions.astype(np.float32),
        scales=scales_log,
        rotations=rotations.astype(np.float32),
        colors=colors.astype(np.float32),
        opacities=opacities,
        errors=errors,
    )

    return result


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
