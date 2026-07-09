"""Test pipeline with multiple real room images."""
import asyncio
import json
import websockets
import httpx
import os

BACKEND = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
UPLOADS = os.path.join(os.path.dirname(__file__), "uploads")

ROOM_IMAGES = [
    "living_room_bright.png",
    "living_room_spacious.png",
    "living_room_cozy.png",
]


async def run_pipeline(image_name: str):
    img_path = os.path.join(UPLOADS, image_name)
    print(f"\n{'='*60}")
    print(f"Testing: {image_name}")
    print(f"{'='*60}")

    # Upload
    with open(img_path, "rb") as f:
        r = httpx.post(f"{BACKEND}/api/generate-world", files={"image": f}, timeout=30)
    job_id = r.json()["job_id"]
    print(f"Job: {job_id}")

    # Connect WebSocket
    uri = f"{WS_URL}/ws/pipeline/{job_id}"
    events = []
    try:
        async with websockets.connect(uri) as ws:
            print("Connected. Monitoring...")
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=600)
                data = json.loads(msg)
                events.append(data)
                evt = data.get("event", "?")
                stage = data.get("stage", "?")
                elapsed = data.get("elapsed_sec", "")
                msg_text = data.get("message", data.get("error", data.get("splat_url", "")))
                extra = f" ({elapsed}s)" if elapsed != "" else ""
                print(f"  [{evt}] {stage}{extra} | {str(msg_text)[:100]}")
                if evt in ("pipeline_done", "error"):
                    break
    except asyncio.TimeoutError:
        print("  TIMEOUT")
    except Exception as e:
        print(f"  EXCEPTION: {e}")

    # Summary
    last = events[-1] if events else {}
    splat_url = last.get("splat_url", "")
    if splat_url:
        r2 = httpx.get(f"{BACKEND}{splat_url}", timeout=10)
        print(f"  Splat: {splat_url} -> {r2.status_code}, {len(r2.content)} bytes")
    else:
        print(f"  No splat generated. Last event: {last.get('event')}")

    # Print stage timings
    for e in events:
        if e.get("event") == "stage_done":
            s = e.get("stage", "?")
            t = e.get("elapsed_sec", 0)
            d = e.get("data", {})
            print(f"    {s}: {t}s | {str(d)[:120]}")

    return splat_url


async def main():
    results = {}
    for img in ROOM_IMAGES:
        splat_url = await run_pipeline(img)
        results[img] = splat_url

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for img, url in results.items():
        status = "OK" if url else "FAILED"
        print(f"  {img}: {status} -> {url}")


asyncio.run(main())
