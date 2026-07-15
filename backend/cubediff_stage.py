"""Local single-image to 360 panorama generation with OpenCubeDiff.

The model jointly denoises six cubemap faces and holds the conditioned front
face fixed in latent space.  That is materially more coherent than issuing six
independent image-generation requests.  The released checkpoint is an
unofficial SD 1.5 reimplementation of CubeDiff, so LucidFrame keeps this as a
separate, explicitly generative mode rather than replacing direct SHARP.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
TAG = "[CUBEDIFF]"
MODEL_ID = "hlicai/cubediff-512-imgonly"
FACE_NAMES = ("front", "back", "left", "right", "zenith", "nadir")

_PIPELINE = None
_MODEL_LOCK = threading.Lock()


def _repository_path() -> Path:
    return Path(__file__).resolve().parents[1] / "external" / "open-cubediff"


def _ensure_import_path() -> Path:
    repository = _repository_path()
    if not (repository / "cubediff" / "pipelines" / "pipeline.py").is_file():
        raise RuntimeError(
            "OpenCubeDiff is not installed. Run scripts/install_quality_models.ps1 first."
        )
    path = str(repository)
    if path not in sys.path:
        sys.path.insert(0, path)
    return repository


def is_available() -> bool:
    try:
        _ensure_import_path()
        import py360convert  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def prepare_conditioning_image(image: Image.Image, size: int = 512) -> Image.Image:
    """Fit a normal photograph into a square face without stretching it.

    CubeDiff requires a square 90-degree front face.  Reflection padding keeps
    every source pixel and avoids the black bars that diffusion would otherwise
    interpret as scene content.  Square inputs pass through unchanged apart
    from the final high-quality resize.
    """
    source = np.asarray(image.convert("RGB"))
    height, width = source.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("The source image has no usable pixels.")
    if width > height:
        total = width - height
        top = total // 2
        bottom = total - top
        source = cv2.copyMakeBorder(source, top, bottom, 0, 0, cv2.BORDER_REFLECT_101)
    elif height > width:
        total = height - width
        left = total // 2
        right = total - left
        source = cv2.copyMakeBorder(source, 0, 0, left, right, cv2.BORDER_REFLECT_101)
    interpolation = cv2.INTER_AREA if max(source.shape[:2]) > size else cv2.INTER_LANCZOS4
    square = cv2.resize(source, (size, size), interpolation=interpolation)
    return Image.fromarray(square)


def _load_pipeline(device):
    global _PIPELINE
    with _MODEL_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        _ensure_import_path()
        import torch
        from cubediff.pipelines.pipeline import CubeDiffPipeline

        model_id = os.getenv("CUBEDIFF_MODEL_ID", MODEL_ID).strip() or MODEL_ID
        logger.info("%s Loading %s", TAG, model_id)
        pipeline = CubeDiffPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        # OpenCubeDiff replaces VAE GroupNorm modules after loading.  Explicitly
        # converting the complete patched pipeline prevents mixed FP16/FP32
        # normalization failures on current Diffusers releases.
        pipeline = pipeline.to(device=device, dtype=torch.float16)
        pipeline.set_progress_bar_config(disable=True)
        _PIPELINE = pipeline
        return pipeline


def generate_360_panorama(
    image_path: str | Path,
    output_dir: Path,
    quality_profile: str = "balanced",
) -> tuple[Path, dict[str, object]]:
    """Generate a connected 2K ERP and its six cubemap faces locally."""
    if not is_available():
        raise RuntimeError("CubeDiff is unavailable because its local package, py360convert, or CUDA is missing.")

    import torch
    from torchvision import transforms

    started = time.time()
    output_dir = Path(output_dir)
    face_dir = output_dir / "generated_cubemap"
    face_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    quality_profile = quality_profile if quality_profile in {"balanced", "detail"} else "balanced"
    steps = int(os.getenv("CUBEDIFF_STEPS_DETAIL" if quality_profile == "detail" else "CUBEDIFF_STEPS_BALANCED", "40" if quality_profile == "detail" else "24"))
    steps = max(8, min(60, steps))
    cfg_scale = float(os.getenv("CUBEDIFF_CFG_SCALE", "3.0"))
    seed = int(os.getenv("CUBEDIFF_SEED", "42"))

    pipeline = _load_pipeline(device)
    face_size = int(pipeline.vae.config.sample_size)
    with Image.open(image_path) as source_image:
        source_size = source_image.size
        conditioning = prepare_conditioning_image(source_image, face_size)
    conditioning.save(output_dir / "cubediff_conditioning.png")
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )(conditioning)

    torch.cuda.reset_peak_memory_stats(device)
    with _MODEL_LOCK, torch.inference_mode():
        result = pipeline(
            prompts="",
            conditioning_image=tensor,
            num_inference_steps=steps,
            cfg_scale=cfg_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
        )

    panorama_path = output_dir / "generated_panorama.png"
    Image.fromarray(result.equirectangular).save(panorama_path)
    for name, face in zip(FACE_NAMES, result.faces_cropped):
        Image.fromarray(face).save(face_dir / f"{name}.png")

    report: dict[str, object] = {
        "backend": "OpenCubeDiff image-conditioned six-face diffusion",
        "model": os.getenv("CUBEDIFF_MODEL_ID", MODEL_ID).strip() or MODEL_ID,
        "implementation": "Juan5713/OpenCubeDiff (unofficial CubeDiff reimplementation)",
        "quality_profile": quality_profile,
        "source_resolution": [int(source_size[0]), int(source_size[1])],
        "conditioning_resolution": [face_size, face_size],
        "panorama_resolution": [int(result.equirectangular.shape[1]), int(result.equirectangular.shape[0])],
        "face_count": len(FACE_NAMES),
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "peak_vram_mb": round(float(torch.cuda.max_memory_allocated(device) / 1024**2), 1),
        "elapsed_sec": round(time.time() - started, 3),
        "provenance": "The front face is conditioned by the upload. All other directions are generated hypotheses.",
    }
    (output_dir / "cubediff_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "%s Generated %dx%d panorama in %.1fs (peak %.1f MB)",
        TAG,
        result.equirectangular.shape[1],
        result.equirectangular.shape[0],
        time.time() - started,
        report["peak_vram_mb"],
    )
    return panorama_path, report


def unload() -> None:
    """Release CubeDiff before panoramic depth and SHARP are loaded."""
    global _PIPELINE
    with _MODEL_LOCK:
        _PIPELINE = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
