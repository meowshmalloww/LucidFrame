"""Upload image and immediately connect to WebSocket pipeline."""
import asyncio
import json
import websockets
import httpx
import os

BACKEND = "http://localhost:8000"
WS_URL = "ws://localhost:8000"
TEST_IMG = os.path.join(os.path.dirname(__file__), "uploads", "test_room.png")

async def main():
    # Step 1: Upload image
    print("Uploading test image...")
    with open(TEST_IMG, "rb") as f:
        r = httpx.post(f"{BACKEND}/api/generate-world", files={"image": f}, timeout=30)
    job_id = r.json()["job_id"]
    print(f"Job created: {job_id}")

    # Step 2: Connect WebSocket immediately
    uri = f"{WS_URL}/ws/pipeline/{job_id}"
    print(f"Connecting to {uri}...")
    
    events = []
    try:
        async with websockets.connect(uri) as ws:
            print("Connected! Monitoring pipeline...\n")
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=600)
                data = json.loads(msg)
                events.append(data)
                evt = data.get("event", "?")
                stage = data.get("stage", "?")
                msg_text = data.get("message", data.get("error", data.get("splat_url", "")))
                print(f"  [{evt}] stage={stage} | {msg_text}")
                
                if evt in ("pipeline_done", "error"):
                    break
    except asyncio.TimeoutError:
        print("\nTIMEOUT after 600s")
    except Exception as e:
        print(f"\nEXCEPTION: {e}")
    
    print(f"\n{'='*60}")
    print(f"Total events: {len(events)}")
    for e in events:
        print(f"  - {e.get('event')}: {e.get('stage', '')} {str(e)[:150]}")
    
    # Check if we got a splat URL
    last = events[-1] if events else {}
    if last.get("event") == "pipeline_done":
        splat_url = last.get("splat_url", "")
        print(f"\nSPLAT URL: {splat_url}")
        if splat_url:
            r2 = httpx.get(f"{BACKEND}{splat_url}", timeout=10)
            print(f"Splat download: {r2.status_code}, size={len(r2.content)} bytes")

asyncio.run(main())
