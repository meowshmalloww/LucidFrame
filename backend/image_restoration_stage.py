"""Conservative, optional source restoration before single-image SHARP.

The reconstruction model already runs at a fixed 1536-pixel internal size, so
this stage is not presented as a way to recover ground-truth detail. It uses
the official compact Real-ESRGAN strong/weak denoise interpolation in padded
tiles and blends its result with Lanczos to reduce GAN texture and line warping.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
TAG = "[RESTORE]"
WEAK_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-wdn-x4v3.pth"
)
WEAK_MODEL_NAME = "realesr-general-wdn-x4v3.pth"
STRONG_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-x4v3.pth"
)
STRONG_MODEL_NAME = "realesr-general-x4v3.pth"


def _checkpoint_path(name: str, url: str) -> Path:
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    if not path.exists() or path.stat().st_size < 1_000_000:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("%s Downloading official Real-ESRGAN checkpoint %s", TAG, name)
        torch.hub.download_url_to_file(url, str(path), progress=True)
    return path


def _denoise_strength(source_profile: str = "photo") -> float:
    """Match Real-ESRGAN's official DNI scale: 0 keeps noise, 1 removes it."""
    default = "0.10" if source_profile == "artwork" else "0.55"
    try:
        configured = float(os.getenv("RESTORATION_DENOISE_STRENGTH", default))
    except ValueError:
        configured = float(default)
    return max(0.0, min(1.0, configured))


def _load_restoration_model(source_profile: str = "photo"):
    """Load one compact model with official strong/weak DNI interpolation."""
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    denoise = _denoise_strength(source_profile)
    weak_path = _checkpoint_path(WEAK_MODEL_NAME, WEAK_MODEL_URL)
    if denoise <= 0.001:
        model = ModelLoader().load_from_file(weak_path)
    else:
        strong_path = _checkpoint_path(STRONG_MODEL_NAME, STRONG_MODEL_URL)
        weak_checkpoint = torch.load(weak_path, map_location="cpu", weights_only=True)
        strong_checkpoint = torch.load(strong_path, map_location="cpu", weights_only=True)
        weak_state = weak_checkpoint.get("params_ema", weak_checkpoint.get("params", weak_checkpoint))
        strong_state = strong_checkpoint.get("params_ema", strong_checkpoint.get("params", strong_checkpoint))
        if weak_state.keys() != strong_state.keys():
            raise RuntimeError("Real-ESRGAN denoise checkpoints have incompatible parameters")
        blended_state = {
            key: strong_state[key] * denoise + weak_state[key] * (1.0 - denoise)
            for key in strong_state
        }
        model = ModelLoader().load_from_state_dict(blended_state)
        del weak_checkpoint, strong_checkpoint, weak_state, strong_state, blended_state

    if not isinstance(model, ImageModelDescriptor) or int(model.scale) != 4:
        raise RuntimeError("The Real-ESRGAN checkpoint did not load as a 4x image model")
    return model, denoise


def _target_scale(width: int, height: int, source_profile: str = "photo") -> float:
    """Choose a useful but bounded scale for SHARP's 1536px internal input."""
    short = min(width, height)
    long = max(width, height)
    target_short = 800.0 if source_profile == "artwork" else 1100.0
    if short >= target_short:
        return 1.0
    return max(1.0, min(2.0, target_short / max(short, 1), 4096.0 / max(long, 1)))


def _run_tiled(model, source: np.ndarray, device, tile_size: int = 320, pad: int = 18) -> np.ndarray:
    """Run a 4x image model in padded, seam-free core tiles."""
    import torch

    height, width = source.shape[:2]
    native_scale = int(model.scale)
    output = np.empty((height * native_scale, width * native_scale, 3), dtype=np.uint8)
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        py0, py1 = max(0, y0 - pad), min(height, y1 + pad)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            px0, px1 = max(0, x0 - pad), min(width, x1 + pad)
            tile = np.ascontiguousarray(source[py0:py1, px0:px1])
            tensor = torch.from_numpy(tile).to(device=device, dtype=torch.float32)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).div_(255.0)
            with torch.inference_mode():
                predicted = model(tensor).clamp_(0.0, 1.0)
            predicted = (
                (predicted[0].float() * 255.0)
                .round()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            crop_x0 = (x0 - px0) * native_scale
            crop_y0 = (y0 - py0) * native_scale
            crop_x1 = crop_x0 + (x1 - x0) * native_scale
            crop_y1 = crop_y0 + (y1 - y0) * native_scale
            output[y0 * native_scale:y1 * native_scale, x0 * native_scale:x1 * native_scale] = (
                predicted[crop_y0:crop_y1, crop_x0:crop_x1]
            )
            del tensor, predicted
    return output


def _wrap_panorama_for_tiling(source: np.ndarray, pad: int) -> np.ndarray:
    """Pad an ERP with reflected poles and a horizontally wrapped seam."""
    if pad <= 0:
        return np.ascontiguousarray(source)
    vertical = np.pad(source, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
    return np.ascontiguousarray(
        np.concatenate([vertical[:, -pad:], vertical, vertical[:, :pad]], axis=1)
    )


def enhance_generated_panorama(
    image_path: Path,
    output_dir: Path,
    source_profile: str = "artwork",
    target_width: int = 2560,
) -> tuple[Path, dict[str, object]]:
    """Restore a generated ERP without introducing a left/right tile seam.

    This improves sampling and compression softness before SHARP-360 makes its
    crops. It cannot add measured geometry or make an inconsistent generated
    direction physically correct.
    """
    import torch

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        source_image = opened.convert("RGB")
    width, height = source_image.size
    target_width = max(width, min(3072, int(round(target_width / 32)) * 32))
    target_height = max(1, round(height * target_width / max(width, 1)))
    if target_width == width:
        return image_path, {
            "applied": False,
            "reason": "generated panorama already meets the detail target",
            "source_resolution": [width, height],
            "output_resolution": [width, height],
        }
    if not torch.cuda.is_available():
        raise RuntimeError("Generated panorama restoration needs CUDA")

    source_profile = source_profile if source_profile in {"artwork", "photo"} else "artwork"
    model, denoise = _load_restoration_model(source_profile)
    device = torch.device("cuda")
    model = model.to(device).eval()

    seam_pad = 24
    source = np.asarray(source_image)
    wrapped = _wrap_panorama_for_tiling(source, seam_pad)
    restored_wrapped = _run_tiled(model, wrapped, device)
    native_scale = int(model.scale)
    restored_4x = restored_wrapped[
        seam_pad * native_scale:(seam_pad + height) * native_scale,
        seam_pad * native_scale:(seam_pad + width) * native_scale,
    ]
    target_size = (target_width, target_height)
    learned = Image.fromarray(restored_4x).resize(target_size, Image.Resampling.LANCZOS)
    measured = source_image.resize(target_size, Image.Resampling.LANCZOS)
    learned_blend = 0.32 if source_profile == "artwork" else 0.58
    restored = Image.blend(measured, learned, learned_blend)
    restored_path = output_dir / "restored_generated_panorama.png"
    restored.save(restored_path, "PNG", optimize=True)

    del model, wrapped, restored_wrapped, restored_4x
    torch.cuda.empty_cache()
    report: dict[str, object] = {
        "applied": True,
        "model": "Real-ESRGAN realesr-general-x4v3 with official DNI blending",
        "execution": "horizontally wrapped, reflected-pole padded local tiles",
        "source_resolution": [width, height],
        "output_resolution": [target_width, target_height],
        "learned_blend": learned_blend,
        "denoise_strength": round(denoise, 3),
        "source_profile": source_profile,
        "limitation": "texture restoration cannot recover true unseen geometry or repair scene-level diffusion inconsistency",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "panorama_restoration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info(
        "%s Enhanced generated ERP %dx%d to %dx%d in %.1fs",
        TAG,
        width,
        height,
        target_width,
        target_height,
        time.time() - started,
    )
    return restored_path, report


def restore_for_reconstruction(
    image_path: Path,
    output_dir: Path,
    source_profile: str = "photo",
) -> tuple[Path, dict[str, object]]:
    """Restore a low-resolution source, or return it unchanged when already sufficient."""
    import torch

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        from image_metadata import preserved_image_metadata
        save_metadata = preserved_image_metadata(opened)
        source_image = opened.convert("RGB")
    width, height = source_image.size
    source_profile = source_profile if source_profile in {"artwork", "photo"} else "artwork"
    scale = _target_scale(width, height, source_profile)
    if scale < 1.1:
        report: dict[str, object] = {
            "applied": False,
            "reason": "source resolution already meets the reconstruction input target",
            "source_resolution": [width, height],
            "output_resolution": [width, height],
            "scale": 1.0,
            "source_profile": source_profile,
        }
        (output_dir / "source_restoration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return image_path, report

    if not torch.cuda.is_available():
        raise RuntimeError("Source restoration needs CUDA")
    model, denoise = _load_restoration_model(source_profile)
    device = torch.device("cuda")
    model = model.to(device).eval()

    source = np.asarray(source_image)
    restored_4x = _run_tiled(model, source, device)
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    learned = Image.fromarray(restored_4x).resize(target_size, Image.Resampling.LANCZOS)
    measured = source_image.resize(target_size, Image.Resampling.LANCZOS)
    # The DNI model is intentionally blended back toward measured pixels. This
    # improves compression/noise without treating hallucinated SR texture as
    # factual source evidence.
    learned_blend = 0.34 if source_profile == "artwork" else 0.68
    restored = Image.blend(measured, learned, learned_blend)
    restored_path = output_dir / "restored_input.png"
    restored.save(restored_path, "PNG", optimize=True, **save_metadata)

    del model, restored_4x
    torch.cuda.empty_cache()
    report = {
        "applied": True,
        "model": "Real-ESRGAN realesr-general-x4v3 with official DNI blending",
        "execution": "padded local tiles through Spandrel",
        "source_resolution": [width, height],
        "output_resolution": list(target_size),
        "scale": round(scale, 4),
        "learned_blend": learned_blend,
        "denoise_strength": round(denoise, 3),
        "source_profile": source_profile,
        "treatment": (
            "low-denoise, low-blend enlargement to preserve intentional artwork texture"
            if source_profile == "artwork"
            else "moderate denoise and restoration for photographic compression/noise"
        ),
        "limitation": "restoration can improve degraded pixels but cannot recover ground-truth unseen detail",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "source_restoration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%s Restored %dx%d to %dx%d in %.1fs", TAG, width, height, *target_size, time.time() - started)
    return restored_path, report
