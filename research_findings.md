# Single-Image Gaussian Reconstruction Findings

## Scope and evidence

This document separates the active production path from experimental code. The current default configuration is `RECONSTRUCTION_MODEL=depth_direct`, `MULTIVIEW_ENABLED=off`, `DIFIX_MODE=skip`, and `SPLAT_STITCH=off`.

### Active path

1. `backend/main.py` receives one upload and skips multi-view generation by default.
2. `backend/reconstruction_stage.py` uses Depth Anything V2 Metric Indoor to estimate a single depth map.
3. It back-projects that map with an assumed 60 degree horizontal field of view, turns samples into oriented Gaussian surfels, and adds texture-projected planes for a room shell.
4. `backend/splat_compiler.py` exports 32-byte `.splat` records for `gsplat` in the browser.

This is monocular depth back-projection, not feed-forward LRM reconstruction. It accurately represents only the camera-visible surface. It cannot contain observed geometry behind objects or behind the input camera.

### Inactive or unsafe paths

- **LGM:** The local `lgm_repo` is the upstream LGM checkout, but the app adapter imports non-existent `lgm.network` and `lgm.renderer` modules. Upstream LGM instead uses `core.models.LGM`, a 4-view ImageDream/MVDream layout, ray embeddings, and `forward_gaussians`. Therefore the current LGM path fails and falls back to depth-direct.
- **Multi-view:** `multiview_stage.py` can generate a six-view Zero123++ grid, but the active depth-direct path ignores it. Zero123++ v1.2 publishes fixed poses: azimuths 30/90/150/210/270/330 degrees, elevations alternating 20/-10 degrees, and 30 degree FOV. It is object-centric and cannot be safely treated as room-camera imagery.
- **Optimization:** `optimize_gaussians_gsplat` uses invented circular camera poses, even when its images have no matching poses. Its current loss has L1 plus a mean-brightness term, rather than D-SSIM, no rendered-depth prior, no edge-aware smoothing, and no anisotropy penalty. It is correctly bypassed by `depth_direct`.
- **Difix:** The included SD-Turbo loop is not official Difix3D+. Official Difix3D+ expects registered multi-view images, COLMAP camera data, and a pre-trained 3DGS checkpoint. Running the included loop on unregistered virtual views can introduce offsets.
- **Stitching:** The optional stitch path reconstructs the same input again and shifts it on X by a heuristic transform. It is not pose registration and must remain disabled.

## Root causes of holes, offsets, and stretching

| Artifact | Current direct cause | Consequence |
| --- | --- | --- |
| Holes / empty reverse views | Single depth map only contains the camera-visible surface. The room shell only fills the outer bounds with image-edge textures. | Back sides of foreground objects and out-of-frame space are absent by construction. |
| Apparent offsets / squashing | The back-projection uses one fixed 60 degree FOV and monocular depth has scale uncertainty. No calibration, sparse geometry alignment, or pose supervision is used. | X/Y proportions and relative object placement can be wrong outside the source view. |
| Stretched / needle splats | The experimental optimizer has no scale-ratio loss or pruning for highly anisotropic ellipsoids. It also optimizes against invented poses. | Gaussian axes can grow to explain incompatible pseudo-views. |
| Unstable quality | The output compiler clamps scales, but only at export; it does not provide a geometry-quality gate. | Invalid, oversized, or extremely anisotropic primitives can reach the viewer. |

The `.splat` rotation convention is not a root cause: this application produces `wxyz`, which is the ordering used by the installed `gsplat` serializer.

## Literature and upstream implementation research

### Multi-view priors for holes

- **Zero123++** produces a fixed six-view object orbit. It is appropriate only for isolated object images after foreground segmentation and normalization, not general interiors. Source: https://github.com/SUDO-AI-3D/zero123plus
- **SV3D** is a latent-video model for image-to-orbit multi-view synthesis and reports stronger novel-view / 3D reconstruction performance than prior Zero123 variants. It is still object-orbit focused, so it should be a separately selected object pipeline rather than a room fallback. Source: https://arxiv.org/abs/2403.12008
- **LGM** expects its own consistent four-view ImageDream/MVDream layout plus ray embeddings. Its official inference uses about 10 GB GPU memory, which is above the project 9.8 GB safe budget. Source: https://github.com/3DTopia/LGM

### Depth priors for offsets

- Depth-Regularized Optimization for 3DGS uses a rendered depth map and a weighted dense monocular-depth guide; it explicitly warns that monocular depth requires scale/offset alignment when multiple posed views are available. Source: https://arxiv.org/html/2311.13398
- DN-Splatter uses edge-aware depth and normal priors for indoor scenes. Its conclusion is directly relevant: depth regularization should not smooth across color/depth edges. Source: https://arxiv.org/html/2403.17822v3
- The installed `gsplat` renderer supports expected depth using `render_mode="RGB+ED"`, so depth loss can be implemented only when real or reliably registered camera poses exist. Source: https://docs.gsplat.studio/main/apis/rasterization.html

### Scale regularization for stretching

For a Gaussian with positive linear scale vector `s=(s_x,s_y,s_z)`, define anisotropy

`r = max(s) / max(min(s), epsilon)`.

A practical soft penalty is

`L_aniso = lambda_aniso * mean(relu(log(r) - log(r_max))^2)`.

At export and after model inference, a hard geometry guard should clamp each scale to `[s_min, s_max]` and raise only undersized axes until `r <= r_max`. Raising minor axes preserves projected coverage and avoids creating holes; shrinking only the major axis may reveal gaps. Low-opacity, non-finite, and degenerate Gaussians should be pruned. This is compatible with conventional density control and with TrimGS-style small-scale geometry control.

## Chosen implementation plan

### Immediate, safe changes

1. Add a single `GeometryQualityConfig` read from environment variables so scale caps, anisotropy cap, opacity floor, target density, shell density, and assumed FOV are explicit and tunable.
2. Add a Gaussian sanitation and quality-report step after either depth or LGM inference and before compilation:
   - reject non-finite rows;
   - normalize quaternions;
   - clamp log-scales in linear space;
   - enforce a configurable maximum scale ratio;
   - prune only low-opacity invalid primitives;
   - record quantitative before/after metrics.
3. Use the configurable FOV in depth back-projection. This does not recover calibration automatically, but makes the source of lateral offset controllable and observable.
4. Do not use invented camera poses for depth-direct optimization, repair, or stitching. Gate those paths behind explicit experimental settings rather than treating them as production quality features.
5. Repair LGM integration to the official local API only after a preflight checks dependencies, checkpoint path, and a GPU budget. It remains an object-only mode.

### Deferred requirements for true 360 degree geometry

A one-image depth map cannot prove hidden geometry. To eliminate reverse-view holes rather than disguise them, use one of the following data-supported pipelines:

- **Object mode:** foreground segment -> Zero123++ or SV3D multi-view generation -> correct posed LGM inference -> optional posed 3DGS refinement with RGB, expected-depth, edge-aware smoothness, and anisotropy losses.
- **Room mode:** capture real multi-view video/photos with camera poses, then optimize with calibrated views and depth/normal priors. Do not use object-orbit generators as a room camera substitute.

### Validation criteria

- Static: Python compilation, import checks, frontend build.
- Geometry: no non-finite values; output scale ratio never exceeds `MAX_GAUSSIAN_ASPECT_RATIO`; scale bounds and opacity floors match configuration.
- View stability: render a fixed source camera plus a documented set of virtual views; report alpha coverage and aspect-ratio statistics. These are smoke metrics, not ground-truth novel-view accuracy.
- Quality: compare a curated fixed image set before/after. A real claim that holes are eliminated requires posed multi-view ground truth or a human review of renders; it cannot be demonstrated from the single source image alone.

## Configuration proposed for implementation

```dotenv
RECONSTRUCTION_FOV_DEG=60
GAUSSIAN_MIN_SCALE=0.003
GAUSSIAN_MAX_SCALE=0.35
GAUSSIAN_MAX_ASPECT_RATIO=4.0
GAUSSIAN_MIN_OPACITY=0.05
GAUSSIAN_PRUNE_OPACITY=0.01
```

These defaults are conservative for source-view quality and prevent needle-like splats. They do not invent missing backside geometry.

## User question: 360° panorama vs. multi-angle images

**Recommendation: Multi-angle images (Option B) is fundamentally better.**

### Option A: AI 360° panorama → 3D
- A 360° equirectangular image is a texture on a sphere — no parallax, no real depth
- Standard depth models (Depth Anything V2) are trained on perspective images, not equirectangular — they produce unreliable depth on panoramas
- The 360° seam (left edge meets right) is almost always visible in AI-generated panoramas
- You'd still need monocular depth estimation, landing back at the same single-depth-map limitation

### Option B: Multi-angle images → 3D
- Multiple views at different angles provide parallax = real geometric information
- This is what Zero123++ → LGM does: generate 4-6 consistent views → feed-forward Gaussian prediction
- The challenge is multi-view consistency: AI-generated views won't perfectly agree
- The diffusion repair papers below solve exactly this problem

## ConFixGS-inspired confidence-weighted repair (implemented)

### Papers applied

- **GSFix3D** (arxiv 2508.14717): Don't trust 3DGS in under-observed regions. Render novel views → repair with diffusion → distill back via photometric loss. Key: detect voids (black/empty pixels) as inpainting targets.
- **ConFixGS** (arxiv 2605.09688): Diffusion-repaired images are NOT ground truth. Estimate per-pixel confidence by measuring repair change relative to original. Low-confidence pixels (likely hallucinations) must be suppressed.
- **NVIDIA Fixer**: Practical GSFix3D — operates on rendered images, fixes holes and blurred textures.

### Implementation in `backend/difix_stage.py`

The previous Difix stage blindly trusted diffusion-repaired images as pseudo-GT. This causes hallucinated geometry to corrupt the 3DGS. The fix adds three ConFixGS-inspired steps:

1. **Void detection** (`_detect_voids`): Identifies black/empty pixels in rendered views that indicate missing geometry. These are legitimate inpainting targets.
2. **Per-pixel confidence estimation** (`_compute_repair_confidence`): Measures how much the diffusion repair changed each pixel. High confidence = repair preserved geometry or filled a void. Low confidence = repair drastically altered well-rendered regions (likely hallucination). Uses exponential decay + spatial smoothing.
3. **Confidence-weighted blending** (`_blend_with_confidence`): Mixes repaired and original views per-pixel based on confidence. High-confidence pixels use the repair; low-confidence pixels keep the original rendering.

### New tunable parameters

```dotenv
DIFIX_CONFIDENCE_THRESHOLD=0.35  # pixels below this confidence are suppressed
DIFIX_VOID_THRESHOLD=0.08        # luminance threshold for void detection
DIFIX_CONFIDENCE_DECAY=0.15      # exponential decay rate for confidence
DIFIX_SMOOTH_KERNEL=7            # spatial smoothing kernel size
```

### Debug visualizations

Each Difix cycle now saves:
- `difix_cycle{N}_view{M}_repaired.png` — raw diffusion output
- `difix_cycle{N}_view{M}_blended.png` — confidence-weighted result used for optimization
- `difix_cycle{N}_view{M}_confidence.png` — jet colormap heatmap of per-pixel confidence

### What this does NOT do

- Does not implement full ConFixGS reprojection-based confidence (requires support view matching)
- Does not suppress densification in low-confidence regions (requires optimizer changes)
- Does not use official Difix3D+ (requires COLMAP + registered multi-view input)
- Remains disabled by default (`DIFIX_MODE=skip`) — enable with `DIFIX_MODE=structural`
