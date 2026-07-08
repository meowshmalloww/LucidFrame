# LucidFrame

> Drop any image. Step into the dream.

AI art installation that takes a single image and hallucinates a walkable 3D Gaussian Splat environment from it. Lucid dream aesthetic, not realism.

## Pipeline

1. **VLM API** (GPT-4o Vision) — analyzes what's *in* the image
2. **LLM API** (GPT-4o/Claude) — imagines what's *beyond* the frame
3. **Zero123++** — generates 6 consistent multi-view images
4. **LGM / Depth+gsplat** — reconstructs 3D Gaussians from multi-view images
5. **Difix3D+** (optional) — fixes structural artifacts
6. **gsplat.js** — renders the .splat file in the browser for WASD exploration

## Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
cp .env.template .env     # Fill in your API keys
python main.py            # Starts on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev               # Starts on http://localhost:3000
```

## Environment Variables

See `backend/.env.template`. At minimum, set one of:
- `OPENAI_API_KEY` — for GPT-4o Vision (VLM) + GPT-4o (LLM)
- `ANTHROPIC_API_KEY` — for Claude Vision (VLM) + Claude (LLM)

## Tech Stack

- **Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS, gsplat.js
- **Backend:** FastAPI, Python 3.11+, PyTorch, diffusers, transformers
- **Models:** Zero123++ v1.2, LGM, Depth Anything V2, Difix3D+
- **VRAM:** 12 GB budget with 20% safety margin (9.6 GB usable)

## Architecture

See `TechnicalArchitecture.md` for the full engineering blueprint.
