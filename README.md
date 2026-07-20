# LucidFrame

LucidFrame turns a single image or panorama into an interactive 3D Gaussian Splat scene that can be explored directly in the browser.

Built for Hack the Arts, the project combines generative imaging, neural reconstruction, and real-time rendering to transform flat artwork and photography into spatial experiences.

## Creation modes

- **Image to 3D** — reconstructs a nearby explorable scene from one image with Apple SHARP.
- **Image to 360** — generates a complete panorama with CubeDiff, then converts it into a 360-degree Gaussian scene.
- **Panorama to 360** — converts a panoramic image into an explorable scene with SPAG4D SHARP-360.
- **World Labs** — optionally creates a hosted Marble world using a World Labs API key.

All three local modes run on the user's computer without paid API calls.

## Features

- Local GPU generation
- Image and panorama restoration
- Interactive Gaussian Splat viewer
- Orbit and free-walk controls
- Live generation progress
- Scene library with batch deletion
- Optional World Labs integration

## Technology

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS
- **Viewer:** Three.js and World Labs Spark
- **Backend:** FastAPI, PyTorch, CUDA, Diffusers
- **Reconstruction:** Apple SHARP and SPAG4D SHARP-360
- **Panorama generation:** CubeDiff
- **Restoration:** Real-ESRGAN

## Install

Requirements: Windows, Python 3.11, Node.js, Git, and an NVIDIA CUDA GPU.

```powershell
git clone https://github.com/meowshmalloww/LucidFrame.git
cd LucidFrame
powershell -ExecutionPolicy Bypass -File .\scripts\install_quality_models.ps1

cd frontend
npm install
```

## Run

Start the backend:

```powershell
cd backend
python main.py
```

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project structure

```text
LucidFrame/
|-- backend/     AI generation, reconstruction, API, and scene storage
|-- frontend/    Website, library, settings, and 3D viewer
`-- scripts/     Local model and dependency installation
```

## Core projects

- [Apple SHARP](https://github.com/apple/ml-sharp)
- [SPAG4D](https://github.com/cedarconnor/SPAG4d)
- [CubeDiff](https://arxiv.org/abs/2501.17162)
- [World Labs Spark](https://github.com/sparkjsdev/spark)

## StructureFirst

StructureFirst reuses only LucidFrame's reconstruction functions: Apple SHARP
for photos, SHARP-360 for panoramas, and the `.splat` compiler. It does not reuse
LucidFrame's interface. Multi-photo room alignment is handled by StructureFirst
before the connected Gaussian scene is compiled.
