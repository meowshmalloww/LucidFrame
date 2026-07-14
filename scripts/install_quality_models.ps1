$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$spagDir = Join-Path $repoRoot "external\spag4d"
$sharpDir = Join-Path $spagDir "spag4d\sharp_arch\ml-sharp"

Write-Host "Installing LucidFrame Python dependencies..."
python -m pip install -r (Join-Path $repoRoot "backend\requirements.txt")

if (-not (Test-Path (Join-Path $spagDir ".git"))) {
    New-Item -ItemType Directory -Force (Split-Path $spagDir) | Out-Null
    git clone --filter=blob:none https://github.com/cedarconnor/SPAG4d.git $spagDir
}

if (-not (Test-Path (Join-Path $sharpDir "pyproject.toml"))) {
    throw "SPAG4D checkout does not include its SHARP adapter at $sharpDir"
}

Write-Host "Installing SPAG4D and Apple SHARP adapters..."
python -m pip install -e $spagDir --no-deps
python -m pip install -e $sharpDir --no-deps

Write-Host "Downloading the official SHARP checkpoint into the PyTorch cache..."
python -c "import torch; from sharp.cli.predict import DEFAULT_MODEL_URL; torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)"

Write-Host "Quality reconstruction models are ready."
