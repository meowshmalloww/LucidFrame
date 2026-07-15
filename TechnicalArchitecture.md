# LucidFrame — technical architecture

This document describes the code that currently runs. It supersedes the earlier Flash3D/depth-shell design notes.

## Runtime

- Frontend: Next.js 15, React 19, TypeScript, `gsplat.js`
- Backend: FastAPI, Python 3.11, PyTorch/CUDA
- Tested GPU: RTX 4080 Laptop GPU, 12 GB VRAM
- Transport: multipart upload, REST job creation, WebSocket progress, static `.splat` delivery

## Mode A: one perspective image

1. The backend decodes the source without reducing it to Flash3D's old 384×256 tensor.
2. Apple SHARP estimates focal length when metadata is unavailable and predicts about 1.18 million metric anisotropic Gaussians at 1536×1536 inference resolution.
3. Invalid, behind-camera, and extremely transparent predictions are removed.
4. Linear color is converted to sRGB; WXYZ quaternion orientation is preserved.
5. The compiler writes positions, learned scales, RGBA, and rotations to `.splat`.
6. The viewer starts at the source camera center and permits look-around plus a bounded 35 cm translation volume.

This is nearby-view synthesis. It cannot reconstruct the back side of an object or a room behind the camera.

## Mode B: one panorama

1. The upload is accepted when it is a usable landscape panorama. It does not need an exact 2:1 ratio.
2. A 2:1 working canvas is built without cropping source pixels:
   - 2:1 input is treated as full equirectangular evidence.
   - wider input is treated as vertically cropped 360 evidence and receives low-confidence pole extension.
   - partial panoramas receive reflected side extension marked as unobserved.
3. Overlapping cube views feed the existing Depth Anything V2 stack. Face depths are aligned in overlaps and blended to one equirectangular disparity reference.
4. SPAG4D SHARP-360 predicts four overlapping horizon Gaussian fields on the 12 GB profile. True 2:1 input may also predict zenith and nadir caps.
5. Each face is scale/shift aligned to the shared panorama disparity and merged by SPAG4D.
6. PLY coordinates are reflected from Y-up into the browser's X-right/Y-down/Z-forward convention, including covariance orientation.
7. Invalid, low-opacity, and extreme-radius outliers are removed. The merged scene is normalized to a 4 m median radius so a 48 cm translation creates useful parallax.
8. The compiler keeps learned anisotropic scales up to 10 cm instead of clipping all support at the old 5 cm ceiling.

The old continuous spherical depth lift remains an automatic fallback. Its manifest says `aligned_depth_fallback`; it is never labeled SHARP-360.

## Mode C: one perspective image to a generated 360 world

1. The source is aspect-preserved inside a square cubemap conditioning face using reflected padding rather than stretching or black bars.
2. OpenCubeDiff jointly denoises six 95-degree faces while holding the conditioned front latent fixed. The faces are cropped to 90 degrees and converted to a 2048×1024 equirectangular panorama.
3. CubeDiff is fully unloaded before panoramic depth inference begins.
4. The generated ERP follows the same SHARP-360 alignment path as a measured panorama, including a shared panoramic depth reference, overlapping horizon fields, and zenith/nadir caps.
5. Exact angular ownership prevents neighboring learned faces from ghosting. A sparse depth-aligned spherical underlay sits behind the learned Gaussians to cover boundary and low-opacity pinholes, while source-direction comparison repairs only severe dark highlight outliers.
6. The manifest records the source direction as observed and all unseen directions as generated. If SHARP-360 is unavailable, the named `aligned_depth_fallback` remains available without being mislabeled.

This mode fills angular space around the camera. It materially improves turning and modest movement in every direction, but it does not make deeply occluded geometry factual or guarantee unlimited translation.

## Binary format

Each `.splat` record is 32 bytes:

| Bytes | Value |
|---:|---|
| 12 | XYZ position, float32 |
| 12 | XYZ scale, float32 |
| 4 | RGBA, uint8 |
| 4 | WXYZ rotation, signed-byte encoding |

The format omits spherical-harmonic coefficients. It is efficient in the browser, but it cannot preserve every view-dependent lighting term in a full 3DGS PLY.

## Memory strategy

Only one large model is retained at a time. Image-to-360 generation unloads CubeDiff before panorama depth; panorama depth unloads Depth Anything before SHARP face inference; SHARP is unloaded before compilation. Four horizon faces are the default because six faces plus pole caps approach the practical memory ceiling on this GPU.

## Quality controls

- `SHARP_MIN_OPACITY=0.01`
- `SHARP360_MIN_OPACITY=0.02`
- `SHARP360_ERP_WIDTH=auto` (Balanced 1536; Detail 2048)
- `SHARP360_SIDE_COUNT=auto` (Balanced 4; Detail 6 overlapping horizon views)
- `SHARP360_INCLUDE_CAPS=auto`
- `SHARP360_TARGET_RADIUS=4.0`
- `SPLAT_MAX_SCALE=1.00` (outlier guard; smaller values can cut holes into SHARP wall splats)

Every successful output records a manifest and model-specific JSON quality report. A failed quality backend falls back transparently or fails the job; demo media is disabled by default.

## Viewer controls

Both modes use one custom camera controller, avoiding conflicting orbit and first-person handlers. Drag changes yaw and pitch. The Move control enables keyboard translation without pointer-lock failure. Movement is clamped around the source camera because extrapolating far outside the training view exposes unsupported geometry.

## Reproducibility

Run `scripts/install_quality_models.ps1`, start the backend and frontend, then run:

```powershell
python backend\test_pipeline.py --backend http://127.0.0.1:8000
npm --prefix frontend run build
```

Static verification also includes `python -m compileall -q backend`, `npx tsc --noEmit`, and `npx eslint src`.

## License boundary

SPAG4D core code is MIT. The Apple SHARP checkpoint is released under Apple's non-commercial research model license. The interface and manifests disclose this boundary.
