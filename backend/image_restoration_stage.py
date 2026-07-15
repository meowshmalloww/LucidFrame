"""Conservative, optional source restoration before single-image SHARP.

The reconstruction model already runs at a fixed 1536-pixel internal size, so
this stage is not presented as a way to recover ground-truth detail.  It uses
the official weak-denoise Real-ESRGAN compact checkpoint in padded tiles and
blends its result with a Lanczos resize to reduce GAN texture and line warping.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
TAG = "[RESTORE]"
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.5.0/realesr-general-wdn-x4v3.pth"
)
MODEL_NAME = "realesr-general-wdn-x4v3.pth"


def _checkpoint_path() -> Path:
    import torch

    path = Path(torch.hub.get_dir()) / "checkpoints" / MODEL_NAME
    if not path.exists() or path.stat().st_size < 1_000_000:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("%s Downloading official weak-denoise Real-ESRGAN checkpoint", TAG)
        torch.hub.download_url_to_file(MODEL_URL, str(path), progress=True)
    return path


def _target_scale(width: int, height: int) -> float:
    """Choose a useful but bounded scale for SHARP's 1536px internal input."""
    short = min(width, height)
    long = max(width, height)
    if short >= 1100:
        return 1.0
    return max(1.0, min(2.0, 1100.0 / max(short, 1), 4096.0 / max(long, 1)))


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


def restore_for_reconstruction(image_path: Path, output_dir: Path) -> tuple[Path, dict[str, object]]:
    """Restore a low-resolution source, or return it unchanged when already sufficient."""
    import torch

    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as opened:
        source_image = opened.convert("RGB")
    width, height = source_image.size
    scale = _target_scale(width, height)
    if scale < 1.1:
        report: dict[str, object] = {
            "applied": False,
            "reason": "source resolution already meets the reconstruction input target",
            "source_resolution": [width, height],
            "output_resolution": [width, height],
            "scale": 1.0,
        }
        (output_dir / "source_restoration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return image_path, report

    if not torch.cuda.is_available():
        raise RuntimeError("Source restoration needs CUDA")
    from spandrel import ImageModelDescriptor, ModelLoader

    checkpoint = _checkpoint_path()
    model = ModelLoader().load_from_file(checkpoint)
    if not isinstance(model, ImageModelDescriptor) or int(model.scale) != 4:
        raise RuntimeError("The Real-ESRGAN checkpoint did not load as a 4x image model")
    device = torch.device("cuda")
    model = model.to(device).eval()

    source = np.asarray(source_image)
    restored_4x = _run_tiled(model, source, device)
    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    learned = Image.fromarray(restored_4x).resize(target_size, Image.Resampling.LANCZOS)
    measured = source_image.resize(target_size, Image.Resampling.LANCZOS)
    # The weak-denoise checkpoint is intentionally blended back toward measured
    # pixels. This improves compression/noise without treating hallucinated SR
    # texture as factual source evidence.
    restored = Image.blend(measured, learned, 0.68)
    restored_path = output_dir / "restored_input.png"
    restored.save(restored_path, "PNG", optimize=True)

    del model, restored_4x
    torch.cuda.empty_cache()
    report = {
        "applied": True,
        "model": "Real-ESRGAN realesr-general-wdn-x4v3 (weak denoise)",
        "execution": "padded local tiles through Spandrel",
        "source_resolution": [width, height],
        "output_resolution": list(target_size),
        "scale": round(scale, 4),
        "learned_blend": 0.68,
        "limitation": "restoration can improve degraded pixels but cannot recover ground-truth unseen detail",
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_dir / "source_restoration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("%s Restored %dx%d to %dx%d in %.1fs", TAG, width, height, *target_size, time.time() - started)
    return restored_path, report
