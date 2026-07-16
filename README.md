# LucidFrame

> A photograph is not a window. It is a boundary that can dream back.

LucidFrame is an interactive artwork for Hack the Arts. It turns a photograph or panorama into browser-rendered 3D Gaussian splats, then lets the visitor move through the boundary between observed pixels and learned geometry.

LucidFrame has four explicit creation paths:

- **Local Image to 3D** — Apple SHARP predicts about 1.18 million metric, anisotropic Gaussians from one image at a fixed 1536×1536 inference resolution. It is the highest-quality local default and is intended for nearby view motion.
- **Local Image to 360 Dream** — OpenCubeDiff jointly generates front, back, left, right, ceiling, and floor from one normal image. SPAG4D SHARP-360 then aligns learned Gaussian fields into a full-look-around scene. A sparse depth-aligned underlay closes low-opacity and face-boundary pinholes without replacing the learned 3D geometry. This path is entirely local after its weights are installed and is intentionally separate from direct SHARP.
- **Local Panorama to 3D** — SPAG4D SHARP-360 predicts overlapping perspective Gaussian fields, aligns them with panoramic depth, and merges them into one full-look-around field. A true 2:1 equirectangular image supplies real horizon, zenith, and nadir pixels; cropped panoramas are accepted but their missing pole bands remain inferred.
- **World Labs** — an optional, separately disclosed hosted route that uses the visitor's own API key. The local modes do not require an API key or per-generation payment.

This is real Gaussian reconstruction, not a panorama textured onto a sphere. The previous single-depth panorama shell remains only as an automatic fallback when SHARP-360 is unavailable.

## Honest limits

One photograph cannot recover surfaces it never sees. Direct SHARP gives strong nearby parallax but does not create a back side. Image to 360 Dream generates that missing sphere as an artistic hypothesis; overlapping SHARP fields make every direction navigable near the shared camera center, but deeply occluded surfaces can still thin out under large translation.

Splatting also does not make a low-detail upload magically sharp. Detail mode offers two explicit source treatments. **Artwork** uses low-denoise, low-blend Real-ESRGAN enlargement and skips rewriting paintings that already meet an 800-pixel short-edge target, preserving intentional brush, paper, and canvas texture. **Photo** keeps the stronger compression/noise cleanup. Source blur still limits recoverable texture, and no restorer can recover factual unseen detail.

The local Image to 360 path is experimental. Its installed OpenCubeDiff checkpoint conditions six 512-pixel faces, so a 2048×1024 export does not contain native 2K detail in every direction. More Gaussian points cannot repair a blurry or inconsistent generated back side; the uploaded front is observed and every other direction is an artistic continuation.

The workspace renders with World Labs Spark 2.1. Its radial ordering keeps translucent splats steadier during rotation, and its documented `0.3` covariance pre-filter removes the dark subpixel lattice that appeared in the previous viewport. **HD render** is an optional presentation pass that raises the drawing-buffer resolution and extends Gaussian tails from √8σ to 3σ. It does not add geometry, repair an unseen surface, or modify the saved `.splat` file.

Apple's released SHARP weights are licensed for non-commercial research use. That is suitable for a hackathon prototype, but a commercial release needs a differently licensed model or permission from Apple. SPAG4D's core code is MIT-licensed.

## Install

Prerequisites: Windows, Git, Python 3.11, Node.js, and an NVIDIA CUDA GPU. The tested machine is an RTX 4080 Laptop GPU with 12 GB VRAM.

```powershell
cd C:\Users\wenje\Downloads\LucidFrame
powershell -ExecutionPolicy Bypass -File .\scripts\install_quality_models.ps1

cd frontend
npm install
```

The setup script installs Python packages, checks out SPAG4D and OpenCubeDiff, and caches the official SHARP checkpoint plus the OpenCubeDiff image-conditioned checkpoint. The downloads are about 2.8 GB and 4.3 GB respectively.

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
SHARP360_ERP_WIDTH=auto        # 2048 in both profiles; Detail adds more views
SHARP360_SIDE_COUNT=auto       # Balanced 4; Detail 6 overlapping horizon views
SHARP360_INCLUDE_CAPS=auto
SHARP360_TARGET_RADIUS=4.0
CUBEDIFF_STEPS_BALANCED=24
CUBEDIFF_STEPS_DETAIL=40
CUBEDIFF_CFG_SCALE=3.0
SPLAT_MAX_SCALE=2.00

VLM_PROVIDER=offline
LLM_PROVIDER=offline
ENABLE_DEMO_FALLBACK=off
```

`SHARP360_INCLUDE_CAPS=auto` predicts zenith and nadir only when the source is a true 2:1 equirectangular image. Balanced uses the tested four-view 12 GB profile; Detail automatically uses a 2048-pixel alignment field and six overlapping horizon views. Numeric environment values remain available as expert overrides.

## Tested output

On the development RTX 4080 Laptop GPU:

- 1015×831 Monet painting in Artwork + Detail → restoration correctly skipped, 1,177,194 valid SHARP Gaussians, 37.67 MB `.splat`, and 9.2 seconds of reconstruction. Its measured 58.32° horizontal field of view and valid 100–420 m sky support are preserved by the viewer's 1000 m far plane.
- 2000×953 hotel photograph → 1,176,739 valid SHARP Gaussians, 37.65 MB `.splat`, a measured 66.12° horizontal field of view, and an 8.1 cm/s nearby-motion speed derived from SHARP's official disparity envelope.
- 570×428 painting in Artwork + Detail → low-denoise 1065×800 enlargement in 4.0 seconds without using a cloud API. The Photo treatment was separately A/B tested and is intentionally stronger.

- 1280×720 room image → 1,177,956 valid SHARP Gaussians, 37.7 MB `.splat`, about 21 seconds including cold model load.
- 512×512 room image in Detail → conservative 1024×1024 restoration, 1,173,868 valid SHARP Gaussians, 37.56 MB `.splat`, 20.6 seconds through the live REST/WebSocket workflow. Learned wall axes up to 0.855 m are preserved rather than clipped into holes.
- 1347×447 cropped panorama → 3,791,059 valid merged Gaussians, 121.3 MB `.splat`, about 56 seconds with four horizon faces. The merged field is normalized to a 4 m median radius for useful bounded parallax.
- 512×512 room image → eight-step CubeDiff smoke panorama in 12.3 seconds at 5.9 GB peak VRAM. The final tested Balanced lift produced 3,351,585 Gaussians (including 131,072 sparse coverage Gaussians), a 107.3 MB `.splat`, and completed the reconstruction stage in about 60 seconds on the target RTX 4080 Laptop GPU. Product defaults use 24 diffusion steps; warm end-to-end generation is typically about two minutes, while cold model loading can take longer.

The Library supports per-scene deletion, multi-select, Select all, and exact reclaimed-space reporting. World Labs request/response handling is covered by a mocked contract test, so verification does not spend API credits.

Spark 2.1 was browser-tested against the exact Monet, hotel, and 6.8-million-Gaussian generated-360 outputs. The direct scenes render without the previous dark point lattice; the generated 360 still exposes the blur and cubeface disagreement already present in the diffusion output.

Run the same live API smoke test with:

```powershell
python backend\test_pipeline.py --backend http://127.0.0.1:8000
```

## Architecture

FastAPI owns inference and streams progress over WebSockets. Next.js owns creation, library, settings, and the immersive viewer. World Labs Spark 2.1 rasterizes the final 32-byte-per-Gaussian `.splat` records in WebGL2.

See [CURRENT_PRODUCT.md](CURRENT_PRODUCT.md) for product scope, [TechnicalArchitecture.md](TechnicalArchitecture.md) for the exact pipeline, and [HACKATHON_SUBMISSION.md](HACKATHON_SUBMISSION.md) for submission wording.

## Research and source projects

- [Apple SHARP project](https://apple.github.io/ml-sharp/) and [official implementation](https://github.com/apple/ml-sharp)
- [SPAG4D panorama-to-splat implementation](https://github.com/cedarconnor/SPAG4d)
- [CubeDiff paper](https://arxiv.org/abs/2501.17162) and [OpenCubeDiff implementation](https://github.com/Juan5713/OpenCubeDiff)
- [PanoDreamer layered panorama reconstruction](https://github.com/avinashpaliwal/PanoDreamer)
- [Mip-Splatting anti-aliasing analysis](https://www.cvlibs.net/publications/Yu2024CVPR.pdf)
- [World Labs Spark renderer](https://github.com/sparkjsdev/spark) and [system design](https://sparkjs.dev/docs/system-design/)
- [Scene4U, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_Scene4U_Hierarchical_Layered_3D_Scene_Reconstruction_from_Single_Panoramic_Image_CVPR_2025_paper.pdf)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
