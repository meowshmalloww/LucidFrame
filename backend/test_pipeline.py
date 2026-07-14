"""Manual live API smoke test.

This module is import-safe so pytest collection never starts a generation.
Run it directly after starting the backend.
"""

import argparse
import asyncio
import json
import os
import pathlib

import httpx
import websockets


async def main(backend: str) -> None:
    ws_url = backend.replace("http://", "ws://").replace("https://", "wss://")
    upload_dir = pathlib.Path(__file__).parent / "uploads"
    img_path = next(
        (
            file
            for directory in sorted(upload_dir.iterdir(), reverse=True)
            if directory.is_dir()
            for file in directory.iterdir()
            if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        None,
    )
    if img_path is None:
        raise FileNotFoundError("No image found under backend/uploads")
    print(f"Using: {img_path}")

    with img_path.open("rb") as image:
        response = httpx.post(
            f"{backend}/api/generate-world",
            files={"image": image},
            data={"provider": "local"},
            timeout=30,
        )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    print(f"Job: {job_id}")

    terminal = None
    async with websockets.connect(f"{ws_url}/ws/pipeline/{job_id}") as socket:
        while True:
            terminal = json.loads(await asyncio.wait_for(socket.recv(), timeout=600))
            event = terminal.get("event", "?")
            stage = terminal.get("stage", "?")
            message = terminal.get("message", terminal.get("error", terminal.get("splat_url", "")))
            print(f"  [{event}] {stage}: {message}")
            if event in {"pipeline_done", "error"}:
                break

    if terminal and terminal.get("event") == "pipeline_done":
        splat_url = terminal.get("splat_url", "")
        result = httpx.get(f"{backend}{splat_url}", timeout=30)
        result.raise_for_status()
        print(f"\nSplat: {result.status_code}, {len(result.content) / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default=os.getenv("LUCIDFRAME_BACKEND", "http://127.0.0.1:8000"),
    )
    args = parser.parse_args()
    asyncio.run(main(args.backend.rstrip("/")))
