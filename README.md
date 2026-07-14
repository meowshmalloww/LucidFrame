# LucidFrame

> A photograph is not a window. It is a boundary that can dream back.

LucidFrame is an interactive artwork for Hack the Arts. It turns a photograph or panorama into browser-rendered 3D Gaussian splats, then lets the visitor move through the boundary between observed pixels and learned geometry.

LucidFrame has three explicit creation paths:

- **Local Image to 3D** — Apple SHARP predicts about 1.18 million metric, anisotropic Gaussians from one image at a fixed 1536×1536 inference resolution. It is the highest-quality local default and is intended for nearby view motion.
- **Local Panorama to 3D** — SPAG4D SHARP-360 predicts overlapping perspective Gaussian fields, aligns them with panoramic depth, and merges them into one full-look-around field. A true 2:1 equirectangular image supplies real horizon, zenith, and nadir pixels; cropped panoramas are accepted but their missing pole bands remain inferred.
- **World Labs** — an optional, separately disclosed hosted route that uses the visitor's own API key. The local modes do not require an API key or per-generation payment.

This is real Gaussian reconstruction, not a panorama textured onto a sphere. The previous single-depth panorama shell remains only as an automatic fallback when SHARP-360 is unavailable.

## Honest limits

One photograph cannot recover surfaces it never sees. SHARP gives strong nearby parallax and considerably finer detail than the old 384×256 Flash3D path, but it does not create a factual back side. Likewise, one panorama observes every direction from one camera center but cannot measure geometry hidden behind foreground objects.

Splatting also does not make a low-detail upload magically sharp. The new predictor removes the old low-resolution model bottleneck; source blur and compression still limit recoverable texture.

Apple's released SHARP weights are licensed for non-commercial research use. That is suitable for a hackathon prototype, but a commercial release needs a differently licensed model or permission from Apple. SPAG4D's core code is MIT-licensed.

## Install

Prerequisites: Windows, Git, Python 3.11, Node.js, and an NVIDIA CUDA GPU. The tested machine is an RTX 4080 Laptop GPU with 12 GB VRAM.

```powershell
cd C:\Users\wenje\Downloads\LucidFrame
powershell -ExecutionPolicy Bypass -File .\scripts\install_quality_models.ps1

cd frontend
npm install
```

The setup script installs Python packages, checks out SPAG4D under `external/spag4d`, installs its bundled SHARP adapter, and caches the official SHARP checkpoint. The checkpoint download is about 2.8 GB.

## Run

Terminal 1:

```powershell
cd C:\Users\wenje\Downloads\LucidFrame\backend
python main.py
```

Terminal 2:

```powershell
cd C:\Users\wenje\Downloads\LucidFrame\frontend
npm run dev
```

Open <http://localhost:3000>. If port 8000 is already occupied, stop the old process instead of starting a second backend.

## Configuration

The quality paths are defaults; no `.env` is required.

```dotenv
RECONSTRUCTION_MODEL=sharp
PANORAMA_RECONSTRUCTION_MODEL=sharp360
SHARP360_ERP_WIDTH=1536
SHARP360_SIDE_COUNT=4
SHARP360_INCLUDE_CAPS=auto
SHARP360_TARGET_RADIUS=4.0
SPLAT_MAX_SCALE=0.10

VLM_PROVIDER=offline
LLM_PROVIDER=offline
ENABLE_DEMO_FALLBACK=off
```

`SHARP360_INCLUDE_CAPS=auto` predicts zenith and nadir only when the source is a true 2:1 equirectangular image. Four overlapping horizon views are the tested 12 GB profile; six or eight views use more time and memory.

## Tested output

On the development RTX 4080 Laptop GPU:

- 1280×720 room image → 1,177,956 valid SHARP Gaussians, 37.7 MB `.splat`, about 21 seconds including cold model load.
- 1347×447 cropped panorama → 3,791,059 valid merged Gaussians, 121.3 MB `.splat`, about 56 seconds with four horizon faces. The merged field is normalized to a 4 m median radius for useful bounded parallax.

Run the same live API smoke test with:

```powershell
python backend\test_pipeline.py --backend http://127.0.0.1:8000
```

## Architecture

FastAPI owns inference and streams progress over WebSockets. Next.js owns creation, library, settings, and the immersive viewer. `gsplat.js` rasterizes the final 32-byte-per-Gaussian `.splat` records in WebGL.

See [CURRENT_PRODUCT.md](CURRENT_PRODUCT.md) for product scope, [TechnicalArchitecture.md](TechnicalArchitecture.md) for the exact pipeline, and [HACKATHON_SUBMISSION.md](HACKATHON_SUBMISSION.md) for submission wording.

## Research and source projects

- [Apple SHARP project](https://apple.github.io/ml-sharp/) and [official implementation](https://github.com/apple/ml-sharp)
- [SPAG4D panorama-to-splat implementation](https://github.com/cedarconnor/SPAG4d)
- [Scene4U, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Scene4U_Hierarchical_Layered_3D_Scene_Reconstruction_from_Single_Panoramic_Image_CVPR_2025_paper.pdf)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
