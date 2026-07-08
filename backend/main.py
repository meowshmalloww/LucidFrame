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
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from progress import ProgressBroadcaster
from resource_monitor import get_health, check_all

# ── Setup ───────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lucidframe")

app = FastAPI(title="LucidFrame", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Serve outputs as static files
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Job storage (in-memory for hackathon)
JOBS: dict[str, dict[str, Any]] = {}


# ── REST Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """System resource status with 20% safety margins."""
    return get_health()


@app.post("/api/generate-world")
async def generate_world(image: UploadFile = File(...)):
    """
    Upload an image and start the LucidFrame pipeline.
    Returns a job_id that can be used to connect to the WebSocket.
    """
    job_id = str(uuid.uuid4())[:12]

    # Save uploaded image
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(image.filename).suffix.lower() or ".png"
    image_path = job_dir / f"input{ext}"
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    JOBS[job_id] = {
        "status": "queued",
        "image_path": str(image_path),
        "job_dir": str(job_dir),
        "output_dir": str(OUTPUTS_DIR / job_id),
        "splat_url": None,
        "error": None,
    }

    logger.info("Created job %s for image %s", job_id, image.filename)
    return {"job_id": job_id}


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

    # ── Stage 1a: VLM — Image Understanding ─────────────────────────────────
    await progress.stage_start("vlm", "Analyzing image via VLM API...")
    await progress.stage_progress("vlm", "Sending image to GPT-4o Vision...")

    from vlm_stage import analyze_image
    analysis = await analyze_image(image_path)

    await progress.stage_done("vlm", {"analysis": analysis})

    # ── Stage 1b: LLM — Dream Narrative ─────────────────────────────────────
    await progress.stage_start("llm", "Generating dreamlike narrative...")
    await progress.stage_progress("llm", "Imagining what lies beyond the frame...")

    from llm_stage import generate_dream_prompt
    master_prompt = await generate_dream_prompt(analysis)

    await progress.stage_done("llm", {"prompt": master_prompt})

    # Save the prompt for reference
    (output_dir / "master_prompt.txt").write_text(master_prompt)
    (output_dir / "vlm_analysis.json").write_text(json.dumps(analysis, indent=2))

    # ── Stage 2: Multi-View Hallucination ───────────────────────────────────
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

    # ── Stage 3: 3D Gaussian Reconstruction ─────────────────────────────────
    await progress.stage_start("reconstruction", "Reconstructing 3D Gaussians...")
    await progress.stage_progress("reconstruction", "Loading reconstruction model...")

    from reconstruction_stage import reconstruct, unload as unload_recon
    gaussians = await asyncio.to_thread(
        reconstruct,
        mv_result["view_paths"],
        output_dir,
    )

    if gaussians.count == 0:
        raise RuntimeError(f"Reconstruction failed: {gaussians.errors}")

    await progress.stage_done("reconstruction", {
        "gaussian_count": gaussians.count,
    })

    # Free VRAM
    await asyncio.to_thread(unload_recon)

    # ── Stage 4: Difix3D+ Artifact Fixing (optional) ────────────────────────
    from difix_stage import fix_artifacts, unload as unload_difix
    difix_mode = os.getenv("DIFIX_MODE", "structural").lower()

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

    await progress.stage_done("compile", {"splat_url": splat_url})

    # ── Pipeline Complete ───────────────────────────────────────────────────
    await progress.pipeline_done(splat_url)
    logger.info("Pipeline complete for job %s → %s", job_id, splat_url)


# ── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info("LucidFrame backend starting up...")
    health = get_health()
    if health["gpu"].get("available"):
        logger.info("GPU: %s (%dMB total, %dMB safe budget)",
                     health["gpu"]["name"],
                     health["gpu"]["total_mb"],
                     health["gpu"]["safe_budget_mb"])
    else:
        logger.warning("No GPU available — pipeline will fail on diffusion stages")
    logger.info("Outputs dir: %s", OUTPUTS_DIR)


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
