import sys
sys.path.insert(0, ".")

print("Testing backend imports...")

# Test individual imports
modules = []
try:
    import fastapi
    modules.append("fastapi")
except ImportError as e:
    print(f"FAIL fastapi: {e}")

try:
    import uvicorn
    modules.append("uvicorn")
except ImportError as e:
    print(f"FAIL uvicorn: {e}")

try:
    import torch
    modules.append(f"torch (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"FAIL torch: {e}")

try:
    import diffusers
    modules.append("diffusers")
except ImportError as e:
    print(f"FAIL diffusers: {e}")

try:
    import transformers
    modules.append("transformers")
except ImportError as e:
    print(f"FAIL transformers: {e}")

try:
    import openai
    modules.append("openai")
except ImportError as e:
    print(f"FAIL openai: {e}")

try:
    import anthropic
    modules.append("anthropic")
except ImportError as e:
    print(f"FAIL anthropic: {e}")

try:
    import psutil
    modules.append("psutil")
except ImportError as e:
    print(f"FAIL psutil: {e}")

try:
    import numpy
    modules.append("numpy")
except ImportError as e:
    print(f"FAIL numpy: {e}")

try:
    import PIL
    modules.append("PIL")
except ImportError as e:
    print(f"FAIL PIL: {e}")

try:
    import cv2
    modules.append("cv2")
except ImportError as e:
    print(f"FAIL cv2: {e}")

try:
    import gsplat
    modules.append("gsplat")
except ImportError as e:
    print(f"FAIL gsplat: {e}")

try:
    from main import app
    routes = [r.path for r in app.routes]
    modules.append(f"main (routes: {len(routes)})")
except Exception as e:
    print(f"FAIL main: {e}")

print()
print(f"OK modules: {', '.join(modules)}")
print(f"Total: {len(modules)} modules imported successfully")

# GPU info
try:
    import torch
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("\nNo CUDA GPU detected")
except Exception as e:
    print(f"\nGPU check failed: {e}")
