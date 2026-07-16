$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$spagDir = Join-Path $repoRoot "external\spag4d"
$sharpDir = Join-Path $spagDir "spag4d\sharp_arch\ml-sharp"
$cubeDiffDir = Join-Path $repoRoot "external\open-cubediff"

Write-Host "Installing LucidFrame Python dependencies..."
python -m pip install -r (Join-Path $repoRoot "backend\requirements.txt")

if (-not (Test-Path (Join-Path $spagDir ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $spagDir) | Out-Null
    git clone --filter=blob:none https://github.com/cedarconnor/SPAG4d.git $spagDir
}

if (-not (Test-Path (Join-Path $cubeDiffDir ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $cubeDiffDir) | Out-Null
    git clone --depth 1 https://github.com/Juan5713/OpenCubeDiff.git $cubeDiffDir
}

if (-not (Test-Path (Join-Path $sharpDir "pyproject.toml"))) {
    throw "SPAG4D checkout does not include its SHARP adapter at $sharpDir"
}

Write-Host "Installing SPAG4D and Apple SHARP adapters..."
python -m pip install -e $spagDir --no-deps
python -m pip install -e $sharpDir --no-deps
python -m pip install -e $cubeDiffDir --no-deps

Write-Host "Downloading the official SHARP checkpoint into the PyTorch cache..."
python -c "import torch; from sharp.cli.predict import DEFAULT_MODEL_URL; torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)"

Write-Host "Downloading the compact Real-ESRGAN strong/weak denoise checkpoints..."
python -c "from pathlib import Path; import torch; base='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/'; names=['realesr-general-x4v3.pth','realesr-general-wdn-x4v3.pth']; root=Path(torch.hub.get_dir())/'checkpoints'; root.mkdir(parents=True, exist_ok=True); [(torch.hub.download_url_to_file(base+n, str(root/n), progress=True) if not (root/n).exists() else None) for n in names]"

Write-Host "Downloading the 4.3 GB local OpenCubeDiff image-conditioned checkpoint..."
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hlicai/cubediff-512-imgonly')"

Write-Host "Quality reconstruction models are ready."
