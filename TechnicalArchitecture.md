# LucidFrame — Technical Architecture & Implementation Plan

> **Companion to `ProjectPlan.md`.** This is the engineering blueprint: model choices, VRAM budget, data flow, risk mitigation, and a concrete build schedule. Every decision is grounded in research of the open-source landscape as of July 2026.

---

## 0. Executive Summary

LucidFrame is an AI art installation that takes a single image — a photo, a painting, a historical artifact, anything — and hallucinates the unseen space around it as a walkable 3D environment. The aesthetic is **lucid dream, not realism**. We are not reconstructing what was actually there. We are expanding the art, generating the space you cannot see, like stepping into a dream of the image.

The user drops an image → a cloud **VLM API** analyzes its visual content → a cloud **LLM API** generates a dreamlike narrative prompt → a local multi-view diffusion model hallucinates unseen angles → a feed-forward reconstruction model predicts 3D Gaussians → an optional artifact fixer cleans up broken regions → the final `.splat` file is served to a Next.js + gsplat.js browser viewer for WASD exploration.

**Target hardware:** NVIDIA RTX 4080m Laptop (12 GB VRAM), Alienware m16 R1.
**VRAM budget:** 12 GB × 0.8 = **9.6 GB usable** (20% safety margin).
**Hackathon:** Hack The Arts — "Create art that couldn't exist without technology."

---

## 1. Key Corrections from Original Plan

| Original Assumption | Corrected Reality |
|---------------------|-------------------|
| RTX 4080 16 GB VRAM | **RTX 4080m Laptop, 12 GB VRAM** |
| 100% local, no APIs | **VLM + LLM via cloud API** (GPT-4o / Claude / Gemini). VLM analyzes the image; LLM generates the dreamlike narrative. Local GPU handles diffusion + 3D only. Can swap to local VLM/LLM later. |
| ArtiFixer (16.9B params) | **Difix3D+ (NVIDIA, CVPR 2025 Oral)** — only 8 GB VRAM, single-step diffusion, far more feasible. ArtiFixer does NOT list Ada Lovelace (RTX 40 series) as supported hardware, has no official quantized version, and requires Linux. |
| Photorealistic reconstruction | **Lucid dream aesthetic** — we are expanding the art, not reconstructing reality. Artifacts and hallucinations are features, not bugs. |
| Single splat, done | **Stitchable splats** — generate the initial space, then optionally expand by generating more views from edges and merging. |
| No temporal consistency concerns | **Temporal/spatial consistency is critical** — multi-view frames must be coherent so the 3D space doesn't have jarring discontinuities. |

---

## 2. The VRAM Problem — 12 GB with 20% Safety Margin

### The Core Constraint

The RTX 4080m has **12 GB VRAM**. With 20% safety margin, our budget is **9.6 GB**. The OS and CUDA runtime reserve ~1-1.5 GB for display buffers, leaving roughly **8-8.5 GB for model inference** in practice.

| Model | Params | FP16 VRAM | Fits 9.6 GB budget? | Notes |
|-------|--------|-----------|---------------------|-------|
| VLM (API) | — | 0 GB local | ✅ | Cloud API, zero local VRAM |
| LLM (API) | — | 0 GB local | ✅ | Cloud API, zero local VRAM |
| Zero123++ v1.2 | ~1B | ~5 GB | ✅ | 6 consistent views, 3×2 tiled layout |
| SV3D | ~1.5B | ~6 GB | ✅ | Orbital views with camera control |
| LGM | ~0.5B + backbones | ~10 GB | ⚠️ Tight | Needs FP16 + reduced resolution |
| Splatter Image | ~0.1B | ~2 GB | ✅ | One Gaussian per pixel, ultra-light |
| Difix3D+ | SD-Turbo base | ~8 GB | ✅ | Single-step, fast, 3DGS-compatible |
| Depth Anything V2 | ~0.3B | ~1.5 GB | ✅ | For depth-based fallback path |
| ArtiFixer | 16.9B | ~34 GB FP16 | ❌ | Not feasible. No quant. Ada Lovelace not listed as supported. Linux only. |

### VRAM Management Strategy

**Sequential loading with explicit cleanup.** Only one heavy model in VRAM at a time:

```
Stage 1a: VLM API call              → 0 GB local → (nothing to free)
Stage 1b: LLM API call              → 0 GB local → (nothing to free)
Stage 2:  Multi-View Diffusion      → ~5-6 GB   → del + torch.cuda.empty_cache()
Stage 3:  3DGS Reconstruction       → ~8-10 GB  → del + torch.cuda.empty_cache()
Stage 4:  Difix3D+ (optional)       → ~8 GB     → del + torch.cuda.empty_cache()
```

Between stages, we call `torch.cuda.empty_cache()` and `del` all model references. The FastAPI backend orchestrates this sequentially.

### Why Not ArtiFixer?

Research findings on ArtiFixer (`nvidia/ArtiFixer` on HuggingFace):

1. **16.9 billion parameters** built on Wan2.1-T2V-14B-Diffusers. At FP16 = ~34 GB. No official quantized version exists.
2. **Supported hardware:** NVIDIA Ampere, Hopper, Blackwell. **Ada Lovelace (RTX 40 series) is NOT listed.** This may mean untested, not unsupported, but it's a red flag.
3. **Supported OS:** Linux only. Our dev machine is Windows.
4. **GGUF quantized Wan2.1 14B exists** (Q4_K_M = 9.73 GB) but ArtiFixer is a fine-tuned variant — the generic GGUF won't work directly. You'd need to quantize ArtiFixer's specific weights.
5. **BlockSwap technique** can run 14B GGUF on 12 GB (~7.2 GB with 12 blocks swapped to CPU), but this is for ComfyUI workflows, not programmatic inference via diffusers.

**Verdict: ArtiFixer is not feasible on 12 GB VRAM RTX 4080m.** Difix3D+ is the better choice.

### Why Difix3D+?

Research findings on Difix3D+ (`nvidia/difix` on HuggingFace, `nv-tlabs/Difix3D` on GitHub):

1. **CVPR 2025 Oral & Best Paper Finalist** — peer-reviewed, high quality.
2. **Only 8 GB VRAM minimum** — fits our 9.6 GB budget with margin.
3. **Single-step diffusion** — fine-tuned from SD-Turbo, one pass to fix artifacts. Fast (~0.35s on A100).
4. **Compatible with both NeRF and 3DGS** — exactly our representation.
5. **Two modes:** offline (distill fixed views back into 3D) and online (real-time post-processing at inference).
6. **Has HuggingFace demo** and clean Python API: `DifixPipeline.from_pretrained("nvidia/difix")`.
7. **Also has "Fixer"** — a newer commercially available variant (`nv-tlabs/Fixer` on GitHub).

**How we use it:** Render novel views from "broken" angles of our 3DGS → feed to Difix → get cleaned images → use as pseudo-GT to optimize the Gaussians. Iterate 3-5 times. For the lucid dream aesthetic, we can tune how aggressively Difix fixes things — maybe only fix structural holes (black voids) but preserve dreamlike texture artifacts.

---

## 3. Model Selection — Researched & Justified

### Stage 1a: VLM — Image Understanding (API)

**Primary: GPT-4o Vision API (OpenAI)**

- **Why:** Best-in-class image understanding. $2.50/1M input tokens, $10/1M output. 2-4 second latency. Handles art, photos, historical images, paintings — anything visual. Produces rich, creative descriptions.
- **Usage:** Send the input image with a prompt asking for: era, style, mood, lighting, architecture, color palette, atmosphere, and a description of what's visible in the image.
- **Output:** Structured visual analysis (JSON) — what's *in* the image.
- **Alternatives:**
  - **Claude Sonnet 4.6** ($3/$15 per 1M) — strong at reasoning about images, 200K context.
  - **Gemini 2.0 Flash** — fastest latency, native video understanding, cheapest.
  - **Local fallback (future):** Llama 3.2 11B Vision Q4 (~7 GB VRAM) via llama.cpp. Can swap in later when we want 100% local.

### Stage 1b: LLM — Dream Narrative Generation (API)

**Primary: GPT-4o / Claude Sonnet 4.6**

- **Why:** Takes the VLM's structured visual analysis and generates a dreamlike narrative prompt imagining what's *beyond* the frame.
- **Input:** VLM visual analysis + system prompt instructing dreamlike, imaginative scene extension (not literal description).
- **Output:** A "Master Scene Prompt" — 2-3 sentences of dreamlike scene description for the diffusion model's conditioning.
- **Why separate from VLM:** VLM extracts what's *in* the image; LLM imagines what's *beyond* it. Splitting gives better creative control over the dream aesthetic. The VLM is the "eyes"; the LLM is the "imagination."
- **Alternatives:**
  - **Gemini 2.0 Flash** — fastest, cheapest.
  - **Local fallback (future):** Llama 3.1 8B Q4 (~5 GB VRAM) via llama.cpp.

**API key handling:** Store in `.env` file, never commit. Backend reads `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`, etc.) at runtime.

### Stage 2: Multi-View Hallucination

**Primary: Zero123++ v1.2 (~5 GB VRAM)**

- **Why:** Proven, lightweight, well-documented. Generates 6 consistent views in a single pass using a 3×2 tiled layout that models the joint distribution of multi-view images. ~5 GB VRAM with FP16. Works with diffusers `DiffusionPipeline`. ParallaxVision already has working code for this (`zero123_synth.py`).
- **Output:** 6 views at 320×320 (or 256×256) in a 3×2 grid, split into individual images.
- **Limitation:** Object-centric (trained on Objaverse). For scenes/landscapes, may not be ideal.
- **Code reference:** `C:\Users\wenje\Downloads\ParallaxVision\backend\zero123_synth.py` — working Zero123++ pipeline with lazy loading, grid splitting, and error handling.

**Alternative: SV3D (Stable Video 3D, ~6 GB VRAM)**

- **Why:** Better for scenes (not just objects). Adapts video diffusion for orbital novel view synthesis. Has camera trajectory control (SV3D_p variant). Temporal consistency is built into the video diffusion architecture.
- **Output:** 4-6 orbital views at 256×256 with known camera poses.
- **When to use:** If Zero123++ quality is insufficient for scene-type images (landscapes, interiors, wide shots).

**Temporal Consistency Approach:**

The user raised concerns about frames being consistent. Here's how each model handles it:

- **Zero123++:** Inherent consistency via tiled joint distribution — all 6 views are generated in a single diffusion pass, sharing noise and attention. No temporal module needed.
- **SV3D:** Uses temporal attention layers from video diffusion architecture. Views are frames in a latent video, so temporal coherence is native.
- **Additional control (optional):**
  - **Depth maps as ControlNet conditioning** — generate depth from the input image (Depth Anything V2), use as spatial control for multi-view generation.
  - **Shared latent noise** — initialize all views with the same noise seed, interlock through the diffusion process.
  - **AnimateDiff motion modules** — if we need video-like frame sequences (not just static multi-view), AnimateDiff injects temporal attention into Stable Diffusion. But for our use case, Zero123++/SV3D already handle this.

### Stage 3: 3D Gaussian Reconstruction

**Primary: LGM (Large Multi-View Gaussian Model, ~10 GB VRAM)**

- **Why:** ECCV 2024 Oral. Feed-forward model that takes multi-view images → 3D Gaussians in a single pass. Outputs raw Gaussian parameters (position, rotation, scale, opacity, SH color). No per-scene optimization needed.
- **VRAM concern:** ~10 GB total (including imagedream + mvdream backbones). With our 9.6 GB budget, this is tight.
- **Mitigations:**
  - Use FP16 inference (already default).
  - Reduce input resolution to 256×256.
  - Use `torch.cuda.amp.autocast()` for mixed precision.
  - If still OOM: offload mvdream backbone to CPU after feature extraction.
- **Output:** Gaussian parameters → compile to `.splat` or `.ply` binary.

**Fallback: Depth-Based Reconstruction (~3 GB VRAM)**

If LGM OOMs, use the ParallaxVision approach:
1. Run **Depth Anything V2** on input image → depth map (~1.5 GB VRAM).
2. Back-project pixels to 3D points using camera intrinsics.
3. Convert point cloud to Gaussians (one Gaussian per point, isotropic scale from nearest-neighbor distance).
4. Run **gsplat optimization** (3DGS training loop) using the multi-view images as supervision.
5. **Code reference:** `C:\Users\wenje\Downloads\ParallaxVision\backend\gaussian_splat.py` — full 3DGS training loop with gsplat, including adaptive density control, opacity reset, pruning. Also `image_to_3d_simple.py` for image+depth → point cloud.

**Ultra-light fallback: Splatter Image (~2 GB VRAM)**

- Single-image → 3D Gaussians in one forward pass. One Gaussian per pixel. Very fast, very light.
- Lower quality but guaranteed to fit in VRAM.
- Good for quick prototyping and testing the pipeline end-to-end.

### Stage 4: Artifact Fixing (Optional, Tunable)

**Primary: Difix3D+ (~8 GB VRAM)**

- **Why:** See Section 2. Only 8 GB VRAM, single-step, 3DGS-compatible.
- **Usage for lucid dream aesthetic:**
  - **Structural fix mode:** Only fix black voids / missing geometry holes. Preserve texture artifacts and dreamlike inconsistencies. This keeps the space walkable without making it sterile.
  - **Full fix mode:** Aggressive artifact removal for cleaner look. Use if judges prefer polish.
  - **Skip mode:** Don't run Difix at all. The raw LGM/splat output has a dreamlike, fragmented quality that may actually be more artistic.
- **Pipeline:**
  1. Render virtual camera views from "broken" angles of the current splat.
  2. Feed rendered images + reference image to Difix.
  3. Difix outputs cleaned images.
  4. Use cleaned images as pseudo-GT for 3-5 gsplat optimization iterations.
  5. Export final `.splat`.

**Alternative: GSFix3D (GSFixer + 3D lifting)**

- ICML 2026 / arXiv 2025. Diffusion-guided repair of novel views in 3DGS.
- Code released Nov 2025 on GitHub (`GSFix3D/GSFix3D`).
- Requires scene-specific fine-tuning on captured data — may add latency.
- More research-grade, less turnkey than Difix3D+.

### Stage 5: Browser Rendering

**Primary: gsplat.js (npm: `gsplat`, v1.2.9)**

- **Why:** Hugging Face's open-source WebGL Gaussian Splatting library. MIT license, 0 dependencies, 534 KB unpacked. Supports `.splat` (compact, fast loading) and `.ply` (with SH coefficients for view-dependent color). 4.5K weekly downloads, actively maintained (last published Jul 2025).
- **API:**
  ```typescript
  import * as SPLAT from "gsplat";

  const scene = new SPLAT.Scene();
  const camera = new SPLAT.Camera();
  const renderer = new SPLAT.WebGLRenderer();
  const controls = new SPLAT.OrbitControls(camera, renderer.canvas);

  await SPLAT.Loader.LoadAsync(splatUrl, scene, () => {});

  const frame = () => {
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  ```
- **Note:** `.splat` format does NOT contain SH coefficients — colors are not view-dependent. For view-dependent color, use `.ply` format instead. For our lucid dream aesthetic, flat colors may actually work fine.
- **Controls:** `SPLAT.OrbitControls` for orbit mode. Custom WASD + pointer-lock for first-person exploration.

**Alternative: `@dvt3d/splat-mesh`**

- Three.js plugin for Gaussian Splatting. GPU texture packing, instanced rendering, WebWorker sorting. Designed for 100K-1M+ splats. If we need Three.js integration (e.g., for post-processing effects, particles, fog), this is the choice.

---

## 4. Splat Stitching — Expanding the Dream

The user wants the space to be expandable — not just one static splat, but a walkable environment that extends beyond the initial reconstruction. Here's the approach:

### Level 1: Single Splat (MVP)

Generate one splat from the input image. The user can orbit/walk around the hallucinated 3D scene. The space is finite but complete — no infinite world, just a "room" of hallucinated geometry around the original viewpoint.

### Level 2: Multi-Splat Stitching (Stretch Goal)

To expand the space:

1. **Identify edge views** — render the current splat from angles at the boundary of what's well-reconstructed.
2. **Generate new views** — feed these edge renders back into Zero123++/SV3D as new seed images, generating views of the space beyond.
3. **Reconstruct additional splats** — run LGM on the new multi-view sets to get additional Gaussian clouds.
4. **Stitch** — transform the new splat's coordinates into the original splat's coordinate system using the known camera poses, then concatenate the Gaussian arrays.

**Simple stitching math:**
```
Given splat A (existing) and splat B (new):
  - Camera pose of B's seed view in A's coordinate system: T_AB (4×4 transform)
  - Transform B's Gaussian positions: p_A = T_AB @ [p_B, 1]
  - Transform B's Gaussian rotations: R_A = T_AB[:3,:3] @ R_B
  - Concatenate: all_gaussians = [A_gaussians, transformed_B_gaussians]
  - Export merged .splat file
```

**Research reference:** GStitch (Pattern Recognition journal, 2025) studies spatial-temporal fusion of 3DGS scenes. Graph-GSReg uses 3D scene graphs for registration. For our hackathon, the simple transform + concatenate approach should suffice.

---

## 5. System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                    NEXT.JS FRONTEND (TypeScript)                  │
│                                                                   │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │  Upload   │  │  Dream Log   │  │  gsplat.js WebGL Viewer    │ │
│  │  Drop Zone│  │  (WebSocket  │  │  (Orbit + WASD controls)   │ │
│  │  (drag &  │  │  progress)   │  │  .splat binary → canvas    │ │
│  │  drop)    │  │              │  │  60+ FPS rasterization     │ │
│  └─────┬─────┘  └──────┬───────┘  └────────────▲──────────────┘ │
│        │               │                        │                │
│        │ POST /api/    │ WS /ws/pipeline        │ GET .splat URL  │
│        │ generate      │ (stage events)         │                 │
└────────┼───────────────┼────────────────────────┼────────────────┘
         │               │                        │
┌────────▼───────────────▼────────────────────────┼────────────────┐
│                FASTAPI BACKEND (Python 3.11+)    │                │
│                                                  │                │
│  ┌─────────────────────────────────────────────┐ │                │
│  │          PIPELINE ORCHESTRATOR               │ │                │
│  │  (sequential stages, VRAM manager)           │ │                │
│  │                                              │ │                │
│  │  Stage 1a: VLM API     → visual analysis     │ │                │
│  │  Stage 1b: LLM API     → Master Scene Prompt │ │                │
│  │  Stage 2:  Zero123++/SV3D → 6 multi-view imgs│ │                │
│  │  Stage 3:  LGM/Depth+gsplat → raw Gaussians  │ │                │
│  │  Stage 4:  Difix3D+ (opt) → cleaned views    │ │                │
│  │  Stage 5:  Compile → final .splat file       │ │                │
│  │                                              │ │                │
│  │  Static file server: /outputs/<id>/final.splat              │
│  └─────────────────────────────────────────────┘                │
│                                                                  │
│  VRAM Manager: del + empty_cache() between every stage           │
│  Progress: WebSocket pushes stage events to frontend              │
│  Resource Monitor: GPU/RAM/CPU usage tracking with 20% margins   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow (Detailed)

```
1. User drops image → POST /api/generate-world (multipart form)
2. Backend saves image, creates job_id, opens WebSocket
3. Stage 1a — VLM API:
   a. Send image to GPT-4o Vision API with analysis prompt
   b. Receive: {era, style, mood, lighting, architecture, color_palette, atmosphere, visible_content}
   c. WS event: {stage: "vlm", status: "done", analysis: {...}}
   d. (No local VRAM used — API call only)
4. Stage 1b — LLM API:
   a. Send VLM analysis to GPT-4o / Claude with dreamlike narrative system prompt
   b. Receive: Master Scene Prompt (2-3 sentences of dreamlike scene description)
   c. WS event: {stage: "llm", status: "done", prompt: "..."}
   d. (No local VRAM used — API call only)
5. Stage 2 — Multi-View Hallucination:
   a. Load Zero123++ v1.2 (or SV3D) from diffusers, FP16, to CUDA
   b. Input: original image (resized to 256×256 or 320×320)
   c. Generate 6 consistent views in single diffusion pass
   d. Split 3×2 grid into individual view images
   e. WS event: {stage: "multiview", status: "done", view_count: 6}
   f. del model; torch.cuda.empty_cache()
6. Stage 3 — 3D Gaussian Reconstruction:
   a. Load LGM (imagedream + mvdream + LGM backbone), FP16
   b. Input: 6 multi-view tensors (kept in VRAM, not disk)
   c. Output: Gaussian parameters {position, rotation, scale, opacity, SH}
   d. Compile to intermediate .splat binary
   e. WS event: {stage: "reconstruction", status: "done", gaussian_count: N}
   f. del model; torch.cuda.empty_cache()
   [If LGM OOMs → fallback to Depth Anything V2 + gsplat optimization]
7. Stage 4 — Artifact Fixing (optional, tunable):
   a. Load Difix3D+ pipeline
   b. Place virtual cameras at broken angles
   c. Render current splat → degraded images
   d. Difix fixes → cleaned images
   e. Use cleaned images as pseudo-GT for 3-5 gsplat optimization iterations
   f. WS event: {stage: "difix", status: "done", frames_fixed: N}
   g. del model; torch.cuda.empty_cache()
8. Stage 5 — Compile & Export:
   a. Convert final Gaussian tensors → binary .splat format
   b. Save to /outputs/<job_id>/final.splat
   c. WS event: {stage: "done", splat_url: "/outputs/.../final.splat"}
9. Frontend receives splat_url → gsplat.js loads binary → render loop starts
10. User explores with WASD + mouse in the lucid dream space
```

---

## 6. Project Structure

```
LucidFrame/
├── frontend/                    # Next.js + TypeScript
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx         # Landing / upload page
│   │   │   └── viewer/
│   │   │       └── page.tsx     # 3D viewer page
│   │   ├── components/
│   │   │   ├── UploadDropzone.tsx    # Drag-and-drop image upload
│   │   │   ├── DreamLog.tsx          # Live WebSocket progress feed
│   │   │   ├── SplatViewer.tsx       # gsplat.js canvas wrapper
│   │   │   ├── Controls.tsx          # WASD + orbit toggle
│   │   │   └── StageProgress.tsx     # Visual pipeline stage indicator
│   │   ├── lib/
│   │   │   ├── api.ts               # REST + WebSocket client
│   │   │   └── usePipeline.ts       # React hook for pipeline state
│   │   └── styles/
│   │       └── globals.css          # Tailwind dark theme
│   ├── package.json
│   ├── tailwind.config.ts
│   └── next.config.ts
│
├── backend/                     # FastAPI (Python 3.11+)
│   ├── main.py                  # FastAPI app, endpoints, WebSocket
│   ├── vlm_stage.py             # Stage 1a: VLM API → visual analysis
│   ├── llm_stage.py             # Stage 1b: LLM API → Master Scene Prompt
│   ├── multiview_stage.py       # Stage 2: Zero123++/SV3D → 6 views
│   ├── reconstruction_stage.py  # Stage 3: LGM or depth+gsplat → Gaussians
│   ├── difix_stage.py           # Stage 4: Difix3D+ → cleaned views (optional)
│   ├── splat_compiler.py        # Gaussian tensor → binary .splat format
│   ├── splat_stitcher.py        # Merge multiple splats (stretch goal)
│   ├── vram_manager.py          # Model load/unload, VRAM monitoring
│   ├── resource_monitor.py      # GPU/RAM/CPU usage with 20% safety margins
│   ├── progress.py              # WebSocket event broadcaster
│   ├── requirements.txt
│   └── .env.template            # API keys, model paths
│
├── TechnicalArchitecture.md     # This document
├── ProjectPlan.md               # Original concept document
└── README.md                    # Setup & run instructions (to be written)
```

---

## 7. API Contracts

### REST Endpoints

```
POST /api/generate-world
  Body: multipart/form-data { image: File }
  Response: { job_id: string }

GET /outputs/{job_id}/final.splat
  Response: binary .splat file (application/octet-stream)

GET /health
  Response: {
    status: "ok",
    gpu: { name, vram_total_mb, vram_free_mb, vram_safe_budget_mb },
    ram: { total_gb, used_gb, safe },
    cpu: { percent, safe },
    models_cached: [...]
  }
```

### WebSocket Events

```
WS /ws/pipeline/{job_id}

→ Server sends:
{ event: "stage_start",    stage: "vlm",            timestamp: ... }
{ event: "stage_progress", stage: "vlm",            message: "Analyzing image via GPT-4o Vision..." }
{ event: "stage_done",     stage: "vlm",            data: { analysis: {...} } }
{ event: "stage_start",    stage: "llm" }
{ event: "stage_progress", stage: "llm",            message: "Generating dreamlike narrative..." }
{ event: "stage_done",     stage: "llm",            data: { prompt: "..." } }
{ event: "stage_start",    stage: "multiview" }
{ event: "stage_progress", stage: "multiview",      message: "Generating 6 views..." }
{ event: "stage_done",     stage: "multiview",      data: { view_count: 6 } }
{ event: "stage_start",    stage: "reconstruction" }
{ event: "stage_progress", stage: "reconstruction", message: "Predicting Gaussians..." }
{ event: "stage_done",     stage: "reconstruction", data: { gaussian_count: N } }
{ event: "stage_start",    stage: "difix" }
{ event: "stage_progress", stage: "difix",          message: "Fixing frame 2/5..." }
{ event: "stage_done",     stage: "difix",          data: { frames_fixed: 5 } }
{ event: "stage_done",     stage: "compile",        data: { splat_url: "/outputs/.../final.splat" } }
{ event: "pipeline_done",  splat_url: "/outputs/.../final.splat" }
{ event: "error",          stage: "...", error: "..." }
```

---

## 8. The .splat Binary Format

The standard 3D Gaussian Splatting `.splat` file format (used by gsplat.js) is a flat binary array. Each Gaussian occupies **32 bytes**:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 12 | Position | x, y, z (float32 each) |
| 12 | 12 | Scale | sx, sy, sz (float32 each) |
| 24 | 4 | Color | r, g, b, a (uint8 each) |
| 28 | 4 | Rotation | quat w, x, y, z (uint8, normalized from float) |

**Note:** `.splat` format does NOT contain SH coefficients — colors are not view-dependent. For view-dependent color, use `.ply` format (gsplat.js supports both).

**Compilation step** (`splat_compiler.py`):
- Position: direct float32 → bytes
- Scale: exponentiate (models output log-scale) → clamp → float32 → bytes
- Color: evaluate SH at default viewing direction → clamp to [0,255] → uint8
- Rotation: normalize quaternion → map from [-1,1] to [0,255] → uint8
- Opacity: sigmoid → map to [0,255] → uint8

**ParallaxVision reference:** `C:\Users\wenje\Downloads\ParallaxVision\backend\gaussian_splat.py` has `_save_gaussians_ply()` (lines 120-163) which writes the PLY format with full Gaussian parameters. We adapt this for `.splat` format.

---

## 9. Resource Management — 20% Safety Margins

The user requires 20% empty safe space for VRAM, RAM, CPU, and GPU usage.

### `resource_monitor.py`

```python
# Pseudocode for resource monitoring
SAFE_MARGIN = 0.20  # 20% safety margin

def get_vram_budget():
    total = torch.cuda.get_device_properties(0).total_memory  # bytes
    usable = total * (1 - SAFE_MARGIN)
    return total, usable

def check_vram_safe(current_usage):
    total, budget = get_vram_budget()
    if current_usage > budget:
        raise VRAMOverheadError(f"Using {current_usage}B, budget is {budget}B")

def get_ram_status():
    import psutil
    mem = psutil.virtual_memory()
    safe_threshold = mem.total * (1 - SAFE_MARGIN)
    return {
        "total_gb": mem.total / 1e9,
        "used_gb": mem.used / 1e9,
        "safe": mem.used < safe_threshold
    }

def get_cpu_status():
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)
    safe_threshold = 100 * (1 - SAFE_MARGIN)  # 80%
    return { "percent": cpu_percent, "safe": cpu_percent < safe_threshold }
```

Before each stage, the backend checks resource availability. If any resource exceeds its safe threshold, the pipeline pauses and sends a WebSocket warning before proceeding with reduced settings (lower resolution, fewer views, skip optional stages).

---

## 10. Frontend Design

### Visual Identity

- **Theme:** Dark, minimalist, dreamlike. Black void background, subtle glow effects, soft particle ambience.
- **Color palette:** `#0a0a0a` (bg), `#1a1a2e` (panels), `#e94560` (accent), `#0f3460` (secondary), `#16c79a` (success).
- **Typography:** Inter for UI, JetBrains Mono for the dream log.

### Key Components

**UploadDropzone:** Full-screen drag-and-drop zone with a pulsing border. Accepts any image — photos, paintings, historical artifacts, screenshots. On drop, shows thumbnail + "Begin Dream" button.

**DreamLog:** Terminal-style log streaming WebSocket events. Shows each stage, progress messages, VLM visual analysis, LLM dream narrative, multi-view generation progress, Gaussian counts, timing. This makes the invisible compute visible — the AI's "dreaming" process as performance art. This is our equivalent of ParallaxVision's provenance sidebar — **the hero UI element that tells the story**.

**SplatViewer:** gsplat.js canvas. Loads `.splat` binary, starts render loop. Camera starts at the original image's viewpoint. User presses WASD to "step through the frame" into the dream.

**StageProgress:** Horizontal progress bar with 6 segments (VLM → LLM → Multi-View → Reconstruction → Difix → Compile). Each segment lights up as its stage completes. Shows elapsed time per stage.

**Controls:** Toggle between Orbit mode (mouse drag to rotate) and First-Person mode (WASD + pointer lock). FOV slider. Reset camera button.

### User Experience Flow

1. Dark page, single glowing drop zone: *"Drop any image. Step into the dream."*
2. User drops image (photo, painting, artifact — anything). "Begin Dream" button glows.
3. Click → split view: left = DreamLog (streaming), right = StageProgress + placeholder for 3D viewer.
4. Stages light up sequentially. Log shows VLM's visual analysis, LLM's dreamlike narrative, multi-view generation progress, Gaussian counts, Difix frame fixing.
5. When pipeline completes → 3D viewer fills the right panel. The original image's viewpoint is the initial camera position.
6. Text overlay: *"Press W to step through the frame."* User walks forward — they're now inside the hallucinated world.
7. Subtle UI fade-out for immersive exploration. Mouse move → look around. WASD → move.

---

## 11. Lessons from ParallaxVision

The reference project (`C:\Users\wenje\Downloads\ParallaxVision`) was a building 3D reconstruction tool. Key lessons:

| ParallaxVision Problem | LucidFrame Fix |
|------------------------|----------------|
| Too many parallel prototypes (6+ codebases, none complete) | **One codebase.** Single `frontend/` + `backend/`. No forks. |
| Scope too ambitious (12-stage pipeline, 20 ML models) | **5 stages, 3-4 models.** VLM API → LLM API → Zero123++ → LGM → Difix (optional). |
| Live ML inference was cut for demo safety (ROCm risk) | **We have CUDA (RTX 4080m).** Live inference is the point. But we still pre-bake fallbacks. |
| No tests, demo broke on stage | **Pre-bake 2-3 sample `.splat` files.** If live pipeline fails, serve cached result with a subtle "cached" indicator. |
| Root README documented APIs that didn't exist | **This doc is the source of truth.** API contracts in Section 7 are what we build to. |
| Provenance sidebar was the hero, not the mesh | **DreamLog is our hero.** The streaming AI thought process is the "art" — it makes compute visible. |

### Reusable Code from ParallaxVision

| File | What it does | How we use it |
|------|-------------|---------------|
| `backend/zero123_synth.py` | Zero123++ pipeline loading, grid splitting, novel view generation | **Direct adaptation** for Stage 2. Already handles lazy loading, FP16, CUDA, error handling. |
| `backend/gaussian_splat.py` | Full 3DGS training loop with gsplat: init from points, Adam optimizer, densification, opacity reset, pruning, PLY export | **Adapt for Stage 3 fallback** (depth+gsplat path) and Stage 4 (Difix pseudo-GT optimization). |
| `backend/image_to_3d_simple.py` | Image + depth → colored 3D point cloud → GLB | **Adapt for depth-based fallback.** Back-projection with bilateral filtering, centering, scaling. |
| `backend/depth_estimator.py` | Depth Anything V2 + Metric3D pipeline | **Adapt for depth fallback.** Already handles FP16, CUDA, multi-scale fusion. |
| `buildvision/backend/main.py` | FastAPI + WebSocket streaming, static file serving | **Pattern reference** for our backend structure. Clean WebSocket event streaming. |
| `buildvision/README.md` | WebSocket event contracts, run instructions | **Pattern reference** for our API contracts. |

---

## 12. Hackathon Strategy

### Demo Plan

1. **Live demo (primary):** Drop a pre-selected image → live pipeline runs → user walks through the frame. ~1.5-3 min compute. DreamLog narrates the AI's process to judges.
2. **Pre-baked fallback (safety):** If pipeline OOMs or crashes, backend serves a pre-computed `.splat` for the same image. DreamLog shows "serving cached dream" briefly.
3. **Gallery mode:** 2-3 pre-computed dream worlds ready to load instantly. Judges explore while live pipeline runs on a new image.

### Pitch Narrative

> "A photograph is a memory frozen in 2D. LucidFrame thaws it. The camera captures one sliver of a moment — everything outside the frame is lost. LucidFrame's AI doesn't just guess what's behind the camera — it hallucinates an entire walkable world from a single image, compiles it into millions of 3D particles, and lets you step through the photograph into the dream. One image in, a world out. Drop a photo. Step inside."

### Multi-Track Pivot Potential

- **Hack The Arts:** "Beyond the Frame" — AI hallucination as artistic medium. Lucid dream aesthetic.
- **ML Empowerment:** "StructureFirst" — heritage preservation from archival photos. Same pipeline, different framing.
- **Open Innovation:** Technical merit — orchestrating bleeding-edge models on 12 GB VRAM with safety margins.

---

## 13. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LGM OOMs on 12 GB VRAM | Medium | High | FP16 + reduced resolution → depth+gsplat fallback → Splatter Image ultra-light |
| Zero123++ poor quality on scenes | Medium | Medium | Switch to SV3D (~6 GB, scene-optimized) |
| Difix3D+ VRAM overlap with LGM | Low | Medium | Sequential loading with empty_cache() — Difix runs AFTER LGM is freed |
| VLM/LLM API latency/down | Low | Medium | Cache common prompts; fallback to Gemini Flash (faster) or local Llama 3.2 Vision / Llama 3.1 8B |
| gsplat.js can't handle large splats | Low | Low | Cap Gaussian count at ~500K; prune low-opacity splats |
| Total pipeline time > 3 min | Medium | Medium | Skip Difix for demo; reduce Zero123++ steps from 75 to 50; pre-bake fallbacks |
| Windows CUDA environment issues | Low | High | Test CUDA 12 + PyTorch setup first; use conda for reproducible env |
| Splat stitching misalignment | Medium | Low (stretch goal) | Simple transform+concatenate first; only needed for Level 2 expansion |
| Demo crash on stage | Medium | Critical | Pre-baked fallback splats; backup screen recording; gallery mode |

---

## 14. Build Schedule

### Phase 1: Foundation (Hours 0-3)
- [ ] Initialize Next.js frontend + FastAPI backend project structure
- [ ] Set up WebSocket pipeline skeleton (frontend connects, receives mock events)
- [ ] Build UploadDropzone + DreamLog UI (streaming mock events)
- [ ] Set up `vram_manager.py` + `resource_monitor.py` with 20% safety margins
- [ ] Verify CUDA 12 + PyTorch + diffusers on RTX 4080m (12 GB)
- [ ] Set up `.env` with VLM + LLM API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)

### Phase 2: VLM + LLM + Multi-View (Hours 3-8)
- [ ] **Stage 1a:** Integrate GPT-4o Vision API → visual analysis JSON
- [ ] **Stage 1b:** Integrate LLM API → Master Scene Prompt (dreamlike prompt engineering)
- [ ] **Stage 2:** Integrate Zero123++ v1.2 → 6 multi-view images
  - Adapt `zero123_synth.py` from ParallaxVision
  - Test with various image types: photos, paintings, historical artifacts
- [ ] Frontend: DreamLog shows VLM analysis + LLM narrative + multi-view previews
- [ ] End-to-end test: image → VLM API → LLM API → Zero123++ → 6 images saved

### Phase 3: 3D Reconstruction (Hours 8-16)
- [ ] **Stage 3 (primary):** Integrate LGM → raw Gaussian tensors
  - FP16, reduced resolution, mixed precision
  - VRAM profiling: ensure peak < 9.6 GB
- [ ] **Stage 3 (fallback):** Depth Anything V2 + gsplat optimization
  - Adapt `gaussian_splat.py` and `image_to_3d_simple.py` from ParallaxVision
- [ ] `splat_compiler.py`: Gaussian tensor → binary .splat format
- [ ] Frontend: gsplat.js viewer loads .splat URL, orbit controls
- [ ] End-to-end test: image → VLM → LLM → Zero123++ → LGM → .splat → browser viewer

### Phase 4: Difix + Polish (Hours 16-22)
- [ ] **Stage 4:** Integrate Difix3D+ → cleaned views → gsplat optimization loop
  - Test structural fix mode vs. full fix mode vs. skip mode
  - Tune for lucid dream aesthetic (preserve some artifacts)
- [ ] First-person WASD controls in gsplat.js viewer
- [ ] DreamLog styling: terminal aesthetic, stage timing, color coding
- [ ] StageProgress component with animated segments
- [ ] Dark theme polish, transitions, loading states

### Phase 5: Demo Prep & Safety (Hours 22-28)
- [ ] Pre-bake 2-3 sample dream worlds (run pipeline on curated images, save .splat files)
- [ ] Fallback logic: if pipeline fails, serve cached .splat
- [ ] Gallery mode: instant-load pre-baked worlds
- [ ] Full end-to-end rehearsal × 5
- [ ] Backup screen recording of complete demo
- [ ] Test on the actual demo machine (Alienware m16 R1, RTX 4080m)
- [ ] Prepare pitch script + demo flow

---

## 15. Dependencies

### Backend (`requirements.txt`)

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9
websockets>=12.0
httpx>=0.27.0
torch>=2.4.0
torchvision>=0.19.0
diffusers>=0.30.0
transformers>=4.44.0
accelerate>=0.33.0
safetensors>=0.4.0
numpy>=1.24.0
Pillow>=10.0.0
opencv-python>=4.9.0
gsplat>=1.0.0
einops>=0.8.0
psutil>=5.9.0
openai>=1.40.0
anthropic>=0.39.0
python-dotenv>=1.0.0
```

### Frontend (`package.json` key deps)

```json
{
  "next": "latest",
  "react": "latest",
  "react-dom": "latest",
  "gsplat": "^1.2.9",
  "tailwindcss": "latest",
  "typescript": "^5.6.0"
}
```

---

## 16. Key Repositories & Models

| Component | Repository | Model Weights |
|-----------|-----------|---------------|
| Difix3D+ | `github.com/nv-tlabs/Difix3D` | `huggingface.co/nvidia/difix` |
| Fixer (newer Difix) | `github.com/nv-tlabs/Fixer` | — |
| Zero123++ | `github.com/SUDO-AI-3D/zero123plus` | `huggingface.co/sudo-ai/zero123plus-v1.2` |
| SV3D | `github.com/Stability-AI/generative-models` | `huggingface.co/stabilityai/stable-video-3d` |
| LGM | `github.com/3DTopia/LGM` | `huggingface.co/3DTopia/LGM` |
| gsplat.js | `github.com/dylanebert/splat.js` | npm: `gsplat` |
| gsplat (CUDA) | `github.com/nerfstudio-project/gsplat` | — (for optimization loop) |
| Depth Anything V2 | — | `huggingface.co/depth-anything/Depth-Anything-V2-Large-hf` |
| GPT-4o Vision (VLM) | `platform.openai.com` | API only |
| GPT-4o / Claude (LLM) | `platform.openai.com` / `anthropic.com` | API only |
| Llama 3.2 Vision (VLM fallback) | `github.com/meta-llama/llama-models` | `huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Llama 3.1 8B (LLM fallback) | `github.com/meta-llama/llama-models` | `huggingface.co/meta-llama/Llama-3.1-8B-Instruct` |
| ArtiFixer (NOT USED) | `github.com/nv-tlabs/ArtiFixer` | `huggingface.co/nvidia/ArtiFixer` — 16.9B, not feasible on 12GB |

---

## 17. Open Questions

1. **LGM on 12 GB:** Will LGM actually run inference without OOM at FP16 with 256×256 input? Needs empirical testing on Day 1. If not, depth+gsplat fallback is ready.
2. **Zero123++ on non-object images:** Zero123++ is trained on Objaverse (objects). How does it handle paintings, landscapes, interior scenes? May need SV3D for these.
3. **Difix3D+ on Windows:** Difix3D+ lists Ampere/Hopper as supported. Ada Lovelace (RTX 40 series) should work (same CUDA compute capability family) but is untested. Needs verification.
4. **Splat stitching coordinate alignment:** When we generate a second splat from edge views, how accurate is the camera pose transform for stitching? May need manual alignment offsets.
5. **gsplat.js + React:** Verify gsplat.js works cleanly with React's concurrent rendering. The `requestAnimationFrame` loop should be fine, but need to verify no lifecycle conflicts.
6. **VLM/LLM prompt engineering:** What prompts produce the best "lucid dream" descriptions? VLM should extract visual facts; LLM should imagine what's beyond the frame. Need to experiment to get dreamlike, imaginative scene extensions rather than literal descriptions.
7. **Total pipeline latency:** Estimate ~1.5-3 min: VLM API (~3s) + LLM API (~3s) + Zero123++ (~30-60s) + LGM (~10-20s) + Difix (~30-60s) + compile (~5s). Is this acceptable for live demo?
