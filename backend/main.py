"""
LucidFrame Backend — FastAPI + WebSocket Pipeline Orchestrator.

Endpoints:
  POST /api/generate-world   — Upload image, start pipeline
  WS   /ws/pipeline/{job_id} — Stream pipeline stage events
  GET  /outputs/{job_id}/final.splat — Download generated .splat file
  GET  /health               — System resource status
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from progress import ProgressBroadcaster
from resource_monitor import get_health, check_all
from scene_camera import load_scene_camera

# ── Setup ───────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lucidframe")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("LucidFrame backend starting up...")
    health = get_health()
    if health["gpu"].get("available"):
        logger.info(
            "GPU: %s (%dMB total, %dMB safe budget)",
            health["gpu"]["name"],
            health["gpu"]["total_mb"],
            health["gpu"]["safe_budget_mb"],
        )
    else:
        logger.warning("No GPU available — local neural reconstruction will be unavailable")
    logger.info("Outputs dir: %s", OUTPUTS_DIR)
    yield


app = FastAPI(title="LucidFrame", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Pre-baked fallback splats (for demo safety)
FALLBACK_SPLAT = OUTPUTS_DIR / "test_galaxy.splat"
# A fallback is only for an intentionally staged offline demo. It must never
# make a failed user generation look like a completed reconstruction.
ENABLE_DEMO_FALLBACK = os.getenv("ENABLE_DEMO_FALLBACK", "off").lower() == "on"

# Serve outputs as static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# Job storage (in-memory for hackathon)
JOBS: dict[str, dict[str, Any]] = {}


class WorldLabsSettingsInput(BaseModel):
    api_key: str = Field(min_length=12, max_length=512)
    model: str = Field(default="marble-1.1", max_length=64)


class GalleryDeleteInput(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=250)


_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _require_loopback(request: Request) -> None:
    """Key configuration is intentionally available only from this computer."""
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="World Labs settings are available only on localhost.")


# ── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """System resource status with 20% safety margins."""
    return get_health()


@app.get("/api/gallery")
async def gallery():
    """List generated splats recursively with manifest and source metadata."""
    splats: list[dict[str, Any]] = []
    for f in OUTPUTS_DIR.rglob("*.splat"):
        relative = f.relative_to(OUTPUTS_DIR)
        if any(part.lower().startswith(("pano-smoke", "test")) for part in relative.parts):
            continue
        # One generated project can contain diagnostic/alternate .splat files.
        # Present its canonical final only so React selection and deletion stay
        # project-based instead of showing duplicate cards with the same ID.
        if f.parent != OUTPUTS_DIR and f.name != "final.splat" and (f.parent / "final.splat").exists():
            continue
        project_relative = f.parent.relative_to(OUTPUTS_DIR)
        job_id = project_relative.as_posix() if f.parent != OUTPUTS_DIR else f.stem
        manifest_path = f.parent / "world_manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                manifest = {}
        source_path = UPLOADS_DIR.joinpath(*PurePosixPath(job_id).parts) / "input.png"
        splats.append({
            "id": job_id,
            "name": str(manifest.get("source_name") or (f.parent.name if f.parent != OUTPUTS_DIR else f.stem)),
            "url": f"/outputs/{relative.as_posix()}",
            "source_url": f"/uploads/{job_id}/input.png" if source_path.exists() else None,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "provider": str(manifest.get("provider") or "local"),
            "coverage": manifest.get("coverage"),
            "camera": manifest.get("camera") or load_scene_camera(f.parent),
            "created_at": f.stat().st_mtime,
        })
    splats.sort(key=lambda item: item["created_at"], reverse=True)
    return {"splats": splats}


def _project_directory(root: Path, job_id: str) -> Path:
    """Resolve a project directory, including legacy nested output groups."""
    relative = PurePosixPath(job_id.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} or not _SAFE_JOB_ID.fullmatch(part)
        for part in relative.parts
    ):
        raise HTTPException(status_code=422, detail=f"Invalid scene id: {job_id!r}")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*relative.parts).resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise HTTPException(status_code=422, detail="Scene id escaped the storage directory.")
    return candidate


@app.get("/api/scenes/{job_id}")
async def scene_metadata(job_id: str):
    """Return renderer metadata for one generated scene."""
    output_dir = _project_directory(OUTPUTS_DIR, job_id)
    manifest_path = output_dir / "world_manifest.json"
    if not output_dir.is_dir() or not manifest_path.is_file():
        raise HTTPException(status_code=404, detail="Scene metadata was not found.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Scene metadata is unreadable.") from exc
    if not manifest.get("camera"):
        manifest["camera"] = load_scene_camera(output_dir)
    return manifest


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _delete_gallery_projects(ids: list[str]) -> dict[str, Any]:
    deleted: list[str] = []
    missing: list[str] = []
    freed_bytes = 0
    for job_id in dict.fromkeys(ids):
        output_dir = _project_directory(OUTPUTS_DIR, job_id)
        upload_dir = _project_directory(UPLOADS_DIR, job_id)
        relative = PurePosixPath(job_id.replace("\\", "/"))
        standalone_splat = OUTPUTS_DIR.resolve() / f"{job_id}.splat" if len(relative.parts) == 1 else None
        existed = output_dir.exists() or upload_dir.exists() or bool(standalone_splat and standalone_splat.is_file())
        if not existed:
            missing.append(job_id)
            continue
        freed_bytes += _directory_size(output_dir) + _directory_size(upload_dir)
        if standalone_splat and standalone_splat.is_file():
            freed_bytes += standalone_splat.stat().st_size
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        if standalone_splat and standalone_splat.is_file():
            standalone_splat.unlink()
        _prune_empty_parents(output_dir.parent, OUTPUTS_DIR)
        _prune_empty_parents(upload_dir.parent, UPLOADS_DIR)
        JOBS.pop(job_id, None)
        deleted.append(job_id)
    return {"deleted": deleted, "missing": missing, "freed_bytes": freed_bytes}


def _prune_empty_parents(start: Path, root: Path) -> None:
    """Remove empty legacy grouping folders without crossing the storage root."""
    resolved_root = root.resolve()
    current = start.resolve()
    while current != resolved_root and resolved_root in current.parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


@app.delete("/api/gallery/{job_id}")
async def delete_gallery_project(job_id: str, request: Request):
    """Delete one local scene and its source upload from this computer."""
    _require_loopback(request)
    return _delete_gallery_projects([job_id])


@app.post("/api/gallery:delete")
async def delete_gallery_projects(payload: GalleryDeleteInput, request: Request):
    """Delete a validated batch of local scenes and report reclaimed bytes."""
    _require_loopback(request)
    return _delete_gallery_projects(payload.ids)


@app.get("/api/providers")
async def providers():
    """Describe the intentionally distinct measured and generated creation paths."""
    from cubediff_stage import is_available as cubediff_available
    from worldlabs_stage import settings_status
    worldlabs = settings_status()
    return {
        "local": {
            "available": True,
            "label": "Local Image to 3D",
            "estimated_time": "about 20 seconds after first model load",
            "description": "Apple SHARP metric Gaussians for high-detail nearby view synthesis.",
            "input": "one normal image",
            "license_note": "SHARP weights are non-commercial research only",
        },
        "local_world": {
            "available": cubediff_available(),
            "label": "Local Image to 360 Dream",
            "estimated_time": "about 2 to 6 minutes depending on model cache state",
            "description": "Joint six-face diffusion followed by depth-aligned SHARP-360 Gaussian reconstruction.",
            "input": "one normal image",
            "license_note": "OpenCubeDiff is an unofficial SD 1.5 reimplementation; SHARP weights are non-commercial research only",
        },
        "local_pano": {
            "available": True,
            "label": "Local Panorama to 3D",
            "estimated_time": "about 1 minute after first model load",
            "description": "Overlapping depth-aligned SHARP views merged into an anisotropic 360 Gaussian scene.",
            "input": "one landscape panorama; 2:1 gives the most complete sphere",
            "license_note": "SPAG4D core is MIT; SHARP weights are non-commercial research only",
        },
        "worldlabs": {
            "available": worldlabs["available"],
            "label": "World Labs Image to World",
            "estimated_time": "about 5 minutes",
            "description": "Hosted 360-degree generative world from one normal image.",
            "requires_credits": True,
            "input": "one normal image",
            "model": worldlabs["model"],
            "estimated_credits": next(option["estimated_credits"] for option in worldlabs["model_options"] if option["id"] == worldlabs["model"]),
            "estimate_label": next(option["estimate_label"] for option in worldlabs["model_options"] if option["id"] == worldlabs["model"]),
        },
    }


@app.get("/api/settings/worldlabs")
async def get_worldlabs_settings(request: Request):
    _require_loopback(request)
    from worldlabs_stage import settings_status
    return settings_status()


@app.post("/api/settings/worldlabs")
async def save_worldlabs_settings(payload: WorldLabsSettingsInput, request: Request):
    _require_loopback(request)
    from worldlabs_stage import WorldLabsError, configure_runtime_settings
    try:
        return await configure_runtime_settings(payload.api_key, payload.model)
    except WorldLabsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/settings/worldlabs")
async def clear_worldlabs_settings(request: Request):
    _require_loopback(request)
    from worldlabs_stage import clear_runtime_settings, settings_status
    clear_runtime_settings()
    return settings_status()

@app.post("/api/generate-world")
async def generate_world(
    image: UploadFile = File(...),
    provider: str = Form("local"),
    creative_direction: str = Form(""),
    quality_profile: str = Form("balanced"),
    source_profile: str = Form("artwork"),
):
    """
    Upload an image and start the LucidFrame pipeline.
    Returns a job_id that can be used to connect to the WebSocket.
    """
    provider = provider.lower().strip()
    if provider not in {"local", "local_world", "local_pano", "worldlabs"}:
        raise HTTPException(status_code=422, detail="Unknown generation mode.")
    quality_profile = quality_profile.lower().strip()
    if quality_profile not in {"balanced", "detail"}:
        raise HTTPException(status_code=422, detail="Unknown reconstruction quality profile.")
    source_profile = source_profile.lower().strip()
    if source_profile not in {"artwork", "photo"}:
        raise HTTPException(status_code=422, detail="Unknown source treatment.")
    if provider == "local_world":
        from cubediff_stage import is_available as cubediff_available

        if not cubediff_available():
            raise HTTPException(
                status_code=503,
                detail="Local Image to 360 Dream needs OpenCubeDiff, py360convert, and CUDA. Run scripts/install_quality_models.ps1 first.",
            )
    if provider == "worldlabs":
        from worldlabs_stage import is_configured, settings_status
        if not is_configured():
            raise HTTPException(status_code=503, detail="World Mode needs a World Labs API key in Settings or backend/.env.")
        worldlabs_model = str(settings_status()["model"])
    else:
        worldlabs_model = None

    job_id = str(uuid.uuid4())[:12]

    # Save uploaded image
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = await image.read()
    ext = Path(image.filename or "upload.png").suffix.lower() or ".png"

    # Normalize to PNG for downstream compatibility. PIL already handles
    # JPEG, PNG, GIF, BMP, TIFF, WebP, AVIF, etc. HEIC/HEIF needs pillow-heif.
    image_path: Path | None = None
    image_size: tuple[int, int] | None = None
    try:
        from PIL import Image
        import io
        pil_img = Image.open(io.BytesIO(raw_bytes))
        from image_metadata import preserved_image_metadata
        save_metadata = preserved_image_metadata(pil_img)
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        image_size = pil_img.size
        image_path = job_dir / "input.png"
        pil_img.save(str(image_path), "PNG", **save_metadata)
        logger.info("Normalized uploaded image %s (%s mode=%s) → PNG", image.filename, ext, pil_img.mode)
    except Exception as pil_exc:
        # Try optional HEIC/HEIF decoder if the extension suggests it
        heic_exts = {".heic", ".heif", ".avci", ".avcs"}
        if ext in heic_exts:
            try:
                import pillow_heif  # noqa: F401
                from PIL import Image
                import io
                pillow_heif.register_heif_opener()
                pillow_heif.register_avif_opener()
                pil_img = Image.open(io.BytesIO(raw_bytes))
                from image_metadata import preserved_image_metadata
                save_metadata = preserved_image_metadata(pil_img)
                if pil_img.mode not in ("RGB", "RGBA"):
                    pil_img = pil_img.convert("RGB")
                image_size = pil_img.size
                image_path = job_dir / "input.png"
                pil_img.save(str(image_path), "PNG", **save_metadata)
                logger.info("Normalized HEIC/HEIF image %s → PNG", image.filename)
            except ImportError:
                logger.error("HEIC/HEIF upload rejected: install pillow-heif (pip install pillow-heif)")
                raise RuntimeError(
                    "HEIC/HEIF images require the optional dependency pillow-heif. "
                    "Install it with: pip install pillow-heif"
                ) from pil_exc
            except Exception as heif_exc:
                logger.error("Failed to decode HEIC/HEIF image %s: %s", image.filename, heif_exc)
                raise RuntimeError(f"Could not decode HEIC/HEIF image {image.filename}") from heif_exc
        else:
            # Fallback: save raw bytes with original extension (downstream may still fail)
            image_path = job_dir / f"input{ext}"
            with open(image_path, "wb") as f:
                f.write(raw_bytes)
            logger.warning("PIL normalization failed (%s) — saved raw bytes as %s", pil_exc, image_path.name)

    if provider == "local_pano":
        if image_size is None:
            raise HTTPException(status_code=422, detail="Local Panorama needs a readable landscape image.")
        from panorama_spherical_stage import PanoramaValidationError, validate_panorama
        try:
            validate_panorama(*image_size)
        except PanoramaValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    JOBS[job_id] = {
        "status": "queued",
        "image_path": str(image_path),
        "job_dir": str(job_dir),
        "output_dir": str(OUTPUTS_DIR / job_id),
        "splat_url": None,
        "error": None,
        "provider": provider,
        "creative_direction": creative_direction.strip()[:240],
        "worldlabs_model": worldlabs_model,
        "quality_profile": quality_profile,
        "source_profile": source_profile,
        "source_name": Path(image.filename or "Untitled").stem[:100],
    }

    logger.info("Created %s job %s for image %s", provider, job_id, image.filename)
    return {
        "job_id": job_id,
        "provider": provider,
        "source_url": f"/uploads/{job_id}/input.png",
    }


# ── WebSocket Pipeline ──────────────────────────────────────────────────────

@app.websocket("/ws/pipeline/{job_id}")
async def pipeline_websocket(ws: WebSocket, job_id: str):
    """
    Stream pipeline stage events to the frontend.

    The frontend connects after receiving a job_id from POST /api/generate-world.
    Events: stage_start, stage_progress, stage_done, pipeline_done, error, warning.
    """
    await ws.accept()

    if job_id not in JOBS:
        await ws.send_json({"event": "error", "error": f"Unknown job_id: {job_id}"})
        await ws.close()
        return

    job = JOBS[job_id]
    job["status"] = "running"

    progress = ProgressBroadcaster(
        send_fn=ws.send_json,
        job_id=job_id,
    )

    try:
        await _run_pipeline(ws, job_id, job, progress)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    except Exception as exc:
        import traceback
        logger.error("Pipeline error for job %s: %s\n%s", job_id, exc, traceback.format_exc())

        # ── Fallback: serve pre-baked .splat if available ────────────────────
        if ENABLE_DEMO_FALLBACK and FALLBACK_SPLAT.exists():
            logger.info("Serving fallback .splat for job %s", job_id)
            fallback_url = f"/outputs/test_galaxy.splat"
            job["splat_url"] = fallback_url
            job["status"] = "done"
            job["fallback"] = True

            await progress.warning("system", f"Pipeline failed; demo fallback was explicitly enabled: {exc}")
            await progress.pipeline_done(fallback_url, {"renderer": "local-splat", "fallback": True})
        else:
            await progress.error("pipeline", str(exc))
            job["status"] = "error"
            job["error"] = str(exc)
    finally:
        if job["status"] != "error":
            job["status"] = "done"


async def _run_pipeline(
    ws: WebSocket,
    job_id: str,
    job: dict[str, Any],
    progress: ProgressBroadcaster,
):
    """Run the full LucidFrame pipeline with progress events."""
    image_path = Path(job["image_path"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resource check ──────────────────────────────────────────────────────
    resource_status = check_all()
    if not resource_status.all_safe:
        for w in resource_status.warnings:
            await progress.warning("system", f"Resource warning: {w}")
    if job.get("provider") == "worldlabs":
        await _run_worldlabs_pipeline(job_id, job, image_path, output_dir, progress)
        return
    if job.get("provider") == "local_pano":
        await _run_local_panorama_pipeline(job_id, job, image_path, output_dir, progress)
        return
    if job.get("provider") == "local_world":
        await _run_local_world_pipeline(job_id, job, image_path, output_dir, progress)
        return

    restoration_report: dict[str, Any] = {"applied": False, "reason": "balanced profile"}
    if job.get("quality_profile") == "detail":
        await progress.stage_start("restoration", "Restoring a low-resolution source in padded local tiles...")
        try:
            from image_restoration_stage import restore_for_reconstruction

            image_path, restoration_report = await asyncio.to_thread(
                restore_for_reconstruction,
                image_path,
                output_dir,
                str(job.get("source_profile") or "artwork"),
            )
            await progress.stage_done("restoration", restoration_report)
        except Exception as exc:
            logger.warning("Source restoration failed for %s: %s", job_id, exc)
            await progress.warning("restoration", f"Source restoration was skipped: {exc}")
            await progress.stage_done("restoration", {"applied": False, "reason": str(exc)})
    else:
        await progress.stage_done("restoration", restoration_report)

    # ── Stage 1a: VLM — Image Understanding ─────────────────────────────────
    await progress.stage_start("vlm", "Reading image composition locally...")
    await progress.stage_progress("vlm", "Keeping the uploaded image on this device...")

    from vlm_stage import analyze_image
    analysis = await analyze_image(image_path)

    await progress.stage_done("vlm", {"analysis": analysis})

    # ── Stage 1b: LLM — Dream Narrative ─────────────────────────────────────
    await progress.stage_start("llm", "Composing an offline dream narrative...")
    await progress.stage_progress("llm", "Turning local visual cues into an art direction...")

    from llm_stage import generate_dream_prompt
    master_prompt = await generate_dream_prompt(analysis)

    await progress.stage_done("llm", {"prompt": master_prompt})

    # Save the prompt for reference
    (output_dir / "master_prompt.txt").write_text(master_prompt)
    (output_dir / "vlm_analysis.json").write_text(json.dumps(analysis, indent=2))

    # Zero123++ normalizes a centred object; it is not a room/scene capture
    # model. Keep it opt-in to avoid fusion tearing on ordinary photographs.
    multiview_enabled = os.getenv("MULTIVIEW_ENABLED", "off").lower() == "on"
    mv_result: dict[str, Any] | None = None

    if multiview_enabled:
        await progress.stage_start("multiview", "Generating 6 multi-view images...")
        await progress.stage_progress("multiview", "Loading Zero123++ diffusion model...")

        from multiview_stage import generate_multiview, unload as unload_multiview
        mv_result = await asyncio.to_thread(
            generate_multiview,
            image_path,
            output_dir / "views",
        )

        if mv_result["errors"]:
            await progress.warning("multiview", f"Errors: {mv_result['errors']}")

        if not mv_result["views"]:
            raise RuntimeError(f"Multi-view generation failed: {mv_result['errors']}")

        await progress.stage_done("multiview", {
            "view_count": len(mv_result["views"]),
            "generation_time_sec": mv_result["generation_time_sec"],
        })

        # Free VRAM
        await asyncio.to_thread(unload_multiview)
    else:
        # Depth-direct mode: skip object-centric multi-view hallucination.
        # The input image alone is enough for monocular depth reconstruction.
        await progress.stage_start("multiview", "Using one measured source frame; no synthetic camera views")
        await progress.stage_done("multiview", {
            "view_count": 1,
            "generation_time_sec": 0.0,
            "skipped": True,
        })

    # ── Stage 3: 3D Gaussian Reconstruction ─────────────────────────────────
    await progress.stage_start("reconstruction", "Reconstructing 3D Gaussians...")
    await progress.stage_progress("reconstruction", "Loading reconstruction model...")

    view_paths = mv_result["view_paths"] if mv_result else [str(image_path)]
    # Prepend the original input image as view 0 for multi-view depth fusion.
    # Zero123++ views are relative to the input, so the input must be first.
    if mv_result and str(image_path) not in view_paths:
        view_paths = [str(image_path)] + view_paths

    from reconstruction_stage import reconstruct, unload as unload_recon
    gaussians = await asyncio.to_thread(
        reconstruct,
        view_paths,
        output_dir,
    )

    if gaussians.count == 0:
        raise RuntimeError(f"Reconstruction failed: {gaussians.errors}")

    await progress.stage_done("reconstruction", {
        "gaussian_count": gaussians.count,
    })

    # Free VRAM
    await asyncio.to_thread(unload_recon)

    # ── Stage 3b: Optional Splat Stitching (expand hallucinated space) ──────
    stitch_mode = os.getenv("SPLAT_STITCH", "off").lower()
    if stitch_mode != "off" and gaussians.count > 0:
        await progress.stage_start("stitch", "Expanding hallucinated space via splat stitching...")
        await progress.stage_progress("stitch", "Generating edge-view splat for stitching...")

        try:
            from splat_stitcher import stitch_gaussians
            from reconstruction_stage import reconstruct_single_depth
            import numpy as np

            # Reconstruct a second splat from a flipped/rotated view
            second_gaussians = await asyncio.to_thread(
                reconstruct_single_depth,
                image_path,
                output_dir / "stitch_views",
            )

            if second_gaussians.count > 0:
                # Transform: offset second cloud by scene extent along X
                extent = float(np.abs(gaussians.positions[:, 0]).max())
                transform = np.eye(4, dtype=np.float32)
                transform[0, 3] = extent * 1.5  # offset along X

                gaussians = stitch_gaussians(gaussians, second_gaussians, transform)
                await progress.stage_done("stitch", {
                    "stitched_count": gaussians.count,
                })
                logger.info("Stitching complete: %d total Gaussians", gaussians.count)
            else:
                await progress.stage_done("stitch", {"stitched_count": 0})
        except Exception as exc:
            logger.warning("Splat stitching failed: %s — skipping", exc)
            await progress.warning("stitch", f"Stitching failed: {exc}")
            await progress.stage_done("stitch", {"stitched_count": 0})

    # ── Stage 4: Difix3D+ Artifact Fixing (optional) ────────────────────────
    from difix_stage import fix_artifacts, unload as unload_difix
    difix_mode = os.getenv("DIFIX_MODE", "skip").lower()

    if difix_mode != "skip":
        await progress.stage_start("difix", f"Fixing artifacts ({difix_mode} mode)...")
        await progress.stage_progress("difix", "Loading Difix3D+ pipeline...")

        gaussians = await asyncio.to_thread(
            fix_artifacts,
            gaussians,
            image_path,
            output_dir,
            difix_mode,
        )

        await progress.stage_done("difix", {"mode": difix_mode})
        await asyncio.to_thread(unload_difix)
    else:
        await progress.stage_done("difix", {"mode": "skipped"})

    # ── Stage 5: Compile & Export ───────────────────────────────────────────
    await progress.stage_start("compile", "Compiling .splat file...")

    from splat_compiler import compile_splat
    splat_path = output_dir / "final.splat"
    await asyncio.to_thread(compile_splat, gaussians, splat_path)

    splat_url = f"/outputs/{job_id}/final.splat"
    job["splat_url"] = splat_url

    reconstruction_model = os.getenv("RECONSTRUCTION_MODEL", "sharp").lower()
    if reconstruction_model == "sharp":
        reconstruction_label = "Apple SHARP high-resolution metric Gaussian reconstruction"
        coverage = "photorealistic nearby views around one measured frame"
    elif reconstruction_model == "flash3d":
        reconstruction_label = "Flash3D feed-forward Gaussian reconstruction"
        coverage = "nearby novel views around one measured frame"
    elif reconstruction_model == "multiview_depth":
        reconstruction_label = "multi-view depth fusion"
        coverage = "generated object views; inspect for synthesis artifacts"
    else:
        reconstruction_label = "monocular depth backprojection"
        coverage = "front view with small parallax"

    manifest = {
        "renderer": "local-splat",
        "provider": "local",
        "source_name": job.get("source_name"),
        "reconstruction_model": reconstruction_model,
        "reconstruction": reconstruction_label,
        "coverage": coverage,
        "artistic_contract": "The source frame is observed; occluded geometry is an artistic continuation, not ground truth.",
        "multiview_enabled": multiview_enabled,
        "difix_mode": difix_mode,
        "quality_profile": job.get("quality_profile", "balanced"),
        "source_profile": job.get("source_profile", "artwork"),
        "source_restoration": restoration_report,
        "camera": load_scene_camera(output_dir),
    }
    (output_dir / "world_manifest.json").write_text(json.dumps(manifest, indent=2))
    await progress.stage_done("compile", {"splat_url": splat_url, **manifest})

    # ── Pipeline Complete ───────────────────────────────────────────────────
    await progress.pipeline_done(splat_url, manifest)
    logger.info("Pipeline complete for job %s → %s", job_id, splat_url)

async def _run_local_world_pipeline(
    job_id: str,
    job: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    progress: ProgressBroadcaster,
) -> None:
    """Dream a full local panorama, then lift it into overlapping learned Gaussians."""
    quality_profile = str(job.get("quality_profile") or "balanced")
    restoration_report: dict[str, Any] = {"applied": False, "reason": "balanced profile"}
    if quality_profile == "detail":
        await progress.stage_start("restoration", "Restoring a low-resolution source before 360 generation...")
        try:
            from image_restoration_stage import restore_for_reconstruction

            image_path, restoration_report = await asyncio.to_thread(
                restore_for_reconstruction,
                image_path,
                output_dir,
                str(job.get("source_profile") or "artwork"),
            )
        except Exception as exc:
            logger.warning("Source restoration failed for generated world %s: %s", job_id, exc)
            restoration_report = {"applied": False, "reason": str(exc)}
            await progress.warning("restoration", f"Source restoration was skipped: {exc}")
    await progress.stage_done("restoration", restoration_report)

    await progress.stage_start("vlm", "Anchoring the uploaded frame as the observed direction...")
    await progress.stage_done(
        "vlm",
        {"source": "local image", "observed_direction": "front", "privacy": "no third-party API"},
    )
    await progress.stage_start("llm", "Using image conditioning instead of a text-only scene guess...")
    await progress.stage_done(
        "llm",
        {"prompt": "Image-only conditioning; the source pixels remain the front-face anchor."},
    )

    await progress.stage_start("multiview", "Dreaming six connected directions on the local GPU...")
    await progress.stage_progress(
        "multiview",
        "Generating front, back, left, right, ceiling, and floor together...",
    )
    from cubediff_stage import generate_360_panorama, unload as unload_cubediff

    try:
        panorama_path, panorama_report = await asyncio.to_thread(
            generate_360_panorama,
            image_path,
            output_dir,
            quality_profile,
        )
    finally:
        await asyncio.to_thread(unload_cubediff)
    panorama_restoration: dict[str, Any] = {
        "applied": False,
        "reason": "balanced profile",
    }
    if quality_profile == "detail":
        await progress.stage_progress(
            "multiview",
            "Restoring the generated sphere with a wrap-aware tiled pass...",
        )
        try:
            from image_restoration_stage import enhance_generated_panorama

            panorama_path, panorama_restoration = await asyncio.to_thread(
                enhance_generated_panorama,
                panorama_path,
                output_dir,
                str(job.get("source_profile") or "artwork"),
                2560,
            )
        except Exception as exc:
            logger.warning("Generated panorama restoration failed for %s: %s", job_id, exc)
            panorama_restoration = {"applied": False, "reason": str(exc)}
            await progress.warning("multiview", f"Panorama restoration was skipped: {exc}")
    await progress.stage_done(
        "multiview",
        {
            "view_count": 6,
            "panorama_resolution": panorama_report.get("panorama_resolution"),
            "elapsed_sec": panorama_report.get("elapsed_sec"),
            "coverage": "generated full sphere",
            "restoration": panorama_restoration,
        },
    )

    await progress.stage_start(
        "reconstruction",
        "Lifting the generated sphere into depth-aligned Gaussian views...",
    )
    await progress.stage_progress(
        "reconstruction",
        "Estimating one shared panoramic depth field before SHARP face prediction...",
    )
    panorama_backend = "sharp360"
    from panorama_spherical_stage import reconstruct_panorama

    try:
        from sharp360_wrapper import reconstruct_sharp360

        gaussians = await asyncio.to_thread(
            reconstruct_sharp360,
            panorama_path,
            output_dir,
            quality_profile,
        )
    except Exception as exc:
        logger.warning(
            "Generated-world SHARP-360 failed for %s: %s; using depth fallback",
            job_id,
            exc,
        )
        await progress.warning(
            "reconstruction",
            f"SHARP-360 unavailable; using aligned depth fallback: {exc}",
        )
        panorama_backend = "aligned_depth_fallback"
        gaussians = await asyncio.to_thread(reconstruct_panorama, panorama_path, output_dir)
    if gaussians.count == 0:
        raise RuntimeError(f"Local 360 world reconstruction failed: {gaussians.errors}")
    await progress.stage_done(
        "reconstruction",
        {
            "gaussian_count": gaussians.count,
            "backend": panorama_backend,
            "coverage": "generated 360 angular coverage with learned nearby-view geometry",
        },
    )

    from reconstruction_stage import unload as unload_recon

    await asyncio.to_thread(unload_recon)
    await progress.stage_start("stitch", "Validating face overlap, scale, and pole coverage...")
    await progress.stage_done(
        "stitch",
        {
            "mode": "joint cubemap plus depth-aligned overlapping Gaussian fields",
            "backend": panorama_backend,
        },
    )
    await progress.stage_done(
        "difix",
        {
            "mode": "not-run",
            "reason": "The six faces share one diffusion process; no independent API images are stitched.",
        },
    )

    await progress.stage_start("compile", "Compiling the generated 360 Gaussian scene...")
    from splat_compiler import compile_splat

    splat_path = output_dir / "final.splat"
    await asyncio.to_thread(compile_splat, gaussians, splat_path)
    splat_url = f"/outputs/{job_id}/final.splat"
    job["splat_url"] = splat_url
    manifest = {
        "renderer": "local-splat",
        "provider": "local_world",
        "source_name": job.get("source_name"),
        "reconstruction": (
            "OpenCubeDiff six-face panorama plus SPAG4D SHARP-360"
            if panorama_backend == "sharp360"
            else "OpenCubeDiff panorama plus aligned depth fallback"
        ),
        "reconstruction_model": panorama_backend,
        "panorama_generator": panorama_report.get("model"),
        "coverage": "generated full 360 sphere: front, back, left, right, ceiling, and floor",
        "artistic_contract": "The uploaded front direction is observed. Every unseen direction and every occluded surface is a generated artistic hypothesis, not recovered ground truth.",
        "translation_note": "Overlapping SHARP fields support nearby movement in every direction. Large translation can still expose surfaces absent from a single camera center.",
        "orientation": "Y-up depth alignment exported as X-right, Y-down, Z-forward browser camera coordinates",
        "quality_profile": quality_profile,
        "source_profile": job.get("source_profile", "artwork"),
        "source_restoration": restoration_report,
        "generation_report": panorama_report,
        "panorama_restoration": panorama_restoration,
    }
    (output_dir / "world_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    await progress.stage_done("compile", {"splat_url": splat_url, **manifest})
    await progress.pipeline_done(splat_url, manifest)
    logger.info("Local generated 360 pipeline complete for job %s", job_id)


async def _run_local_panorama_pipeline(
    job_id: str,
    job: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    progress: ProgressBroadcaster,
) -> None:
    """Build a depth-aligned, multi-face local Gaussian scene from a panorama."""
    from PIL import Image
    from panorama_spherical_stage import reconstruct_panorama, validate_panorama

    with Image.open(image_path) as source:
        validate_panorama(*source.size)

    await progress.stage_done(
        "restoration",
        {"applied": False, "reason": "panorama reconstruction preserves the uploaded ERP pixels"},
    )
    await progress.stage_start("vlm", "Reading the panorama dimensions and projection locally...")
    await progress.stage_done("vlm", {"input_projection": "panoramic", "source": "local"})
    await progress.stage_start("llm", "Keeping the uploaded panorama as the color source...")
    await progress.stage_done("llm", {"prompt": "No image generation is used in the local panorama path."})
    await progress.stage_start("multiview", "Extracting overlapping horizon views and measured pole caps...")
    await progress.stage_done("multiview", {"view_count": "adaptive", "source": "measured panorama crops"})
    await progress.stage_start("reconstruction", "Predicting anisotropic Gaussians and aligning face depths...")
    panorama_backend = os.getenv("PANORAMA_RECONSTRUCTION_MODEL", "sharp360").lower()
    if panorama_backend == "sharp360":
        try:
            from sharp360_wrapper import reconstruct_sharp360

            gaussians = await asyncio.to_thread(
                reconstruct_sharp360,
                image_path,
                output_dir,
                str(job.get("quality_profile") or "balanced"),
            )
        except Exception as exc:
            logger.warning("SHARP-360 failed for %s: %s; using spherical fallback", job_id, exc)
            await progress.warning("reconstruction", f"SHARP-360 unavailable; using depth fallback: {exc}")
            panorama_backend = "aligned_depth_fallback"
            gaussians = await asyncio.to_thread(reconstruct_panorama, image_path, output_dir)
    else:
        panorama_backend = "aligned_depth_fallback"
        gaussians = await asyncio.to_thread(reconstruct_panorama, image_path, output_dir)
    if gaussians.count == 0:
        raise RuntimeError(f"Local panorama reconstruction failed: {gaussians.errors}")
    await progress.stage_done(
        "reconstruction",
        {"gaussian_count": gaussians.count, "coverage": "360 angular coverage with learned local geometry", "backend": panorama_backend},
    )

    from reconstruction_stage import unload as unload_recon
    await asyncio.to_thread(unload_recon)
    await progress.stage_start("stitch", "Checking face overlap, scale, and pole coverage...")
    await progress.stage_done("stitch", {"mode": "depth-aligned overlapping Gaussian faces", "backend": panorama_backend})
    await progress.stage_done("difix", {"mode": "not-run", "reason": "No generated image repair is used in the local panorama path."})
    await progress.stage_start("compile", "Compiling the local panoramic splat...")
    from splat_compiler import compile_splat
    splat_path = output_dir / "final.splat"
    await asyncio.to_thread(compile_splat, gaussians, splat_path)
    splat_url = f"/outputs/{job_id}/final.splat"
    job["splat_url"] = splat_url
    manifest = {
        "renderer": "local-splat",
        "provider": "local_pano",
        "source_name": job.get("source_name"),
        "reconstruction": "SPAG4D SHARP-360 anisotropic multi-face Gaussians" if panorama_backend == "sharp360" else "aligned cube-depth spherical fallback",
        "reconstruction_model": panorama_backend,
        "coverage": "horizontal 360 coverage for full/cropped panoramas; best-effort extension for partial panoramas",
        "artistic_contract": "One panorama observes one camera center. SHARP predicts local geometry, but hidden surfaces behind foreground objects are not measured.",
        "translation_note": "Viewer free-flight is unrestricted. Raw local Gaussians do not include a collision mesh.",
        "orientation": "Y-up depth alignment exported as X-right, Y-down, Z-forward browser camera coordinates",
        "quality_profile": job.get("quality_profile", "balanced"),
    }
    (output_dir / "world_manifest.json").write_text(json.dumps(manifest, indent=2))
    await progress.stage_done("compile", {"splat_url": splat_url, **manifest})
    await progress.pipeline_done(splat_url, manifest)
    logger.info("Local panorama pipeline complete for job %s", job_id)


async def _run_worldlabs_pipeline(
    job_id: str,
    job: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    progress: ProgressBroadcaster,
) -> None:
    """Run the paid 360-degree branch without routing its SPZ output through the local renderer."""
    await progress.stage_done(
        "restoration",
        {"applied": False, "reason": "the hosted provider receives the uploaded source directly"},
    )
    await progress.stage_start("vlm", "Reading image composition before the world expands...")
    from vlm_stage import analyze_image
    analysis = await analyze_image(image_path)
    await progress.stage_done("vlm", {"analysis": analysis})

    await progress.stage_start("llm", "Writing a small direction for the unseen space...")
    from llm_stage import generate_dream_prompt
    master_prompt = await generate_dream_prompt(analysis)
    direction = str(job.get("creative_direction") or "").strip()
    world_prompt = master_prompt if not direction else f"{master_prompt}. Art direction: {direction}."
    await progress.stage_done("llm", {"prompt": world_prompt})
    (output_dir / "master_prompt.txt").write_text(world_prompt)
    (output_dir / "vlm_analysis.json").write_text(json.dumps(analysis, indent=2))

    await progress.stage_start("multiview", "Preparing a 360-degree panoramic continuation...")
    from worldlabs_stage import generate_world

    async def report(message: str) -> None:
        await progress.stage_progress("multiview", message)

    result = await generate_world(
        image_path=image_path,
        display_name=f"LucidFrame - {job_id}",
        text_prompt=world_prompt,
        progress=report,
        model=str(job.get("worldlabs_model") or "marble-1.1"),
    )
    await progress.stage_done("multiview", {"world_id": result["world_id"], "world_url": result["world_url"]})

    await progress.stage_start("reconstruction", "Receiving World Labs native splat world...")
    await progress.stage_done("reconstruction", {"format": "SPZ", "world_url": result["world_url"]})
    await progress.stage_done("stitch", {"mode": "native-world"})
    await progress.stage_done("difix", {"mode": "native-world"})

    await progress.stage_start("compile", "Recording the world's provenance...")
    manifest = {
        "renderer": "worldlabs-native",
        "provider": "worldlabs",
        "input": "single non-panorama image",
        "model": result["model"],
        "world_id": result["world_id"],
        "world_url": result["world_url"],
        "caption": result["caption"],
        "artistic_contract": "World Mode creates a generated 360-degree continuation from one image; it is not a recovered survey of hidden space.",
    }
    (output_dir / "world_manifest.json").write_text(json.dumps(manifest, indent=2))
    job["world_url"] = result["world_url"]
    await progress.stage_done("compile", manifest)
    await progress.pipeline_done("", manifest)
    logger.info("World Labs pipeline complete for job %s -> %s", job_id, result["world_url"])



if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)
