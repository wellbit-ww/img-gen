# Downloads model weights into the named Docker volume "models_data" (mounted at /models).
# Usage:  .\scripts\download_models.ps1            # Qwen-Image-2.1, Turbo, BiRefNet, 4x-UltraSharp
#         .\scripts\download_models.ps1 -WithFlux  # + FLUX klein (gated, needs HF_TOKEN)
param(
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$WithFlux
)

$ErrorActionPreference = "Stop"

docker volume create models_data | Out-Null

# Directory names must match what app/engines expects under MODELS_DIR.
$steps = @(
    "hf download Qwen/Qwen-Image-2.1 --local-dir /models/Qwen__Qwen-Image-2.1",
    "hf download Qwen/Qwen-Image-2.1-Turbo --local-dir /models/Qwen__Qwen-Image-2.1-Turbo",
    "hf download ZhengPeng7/BiRefNet --local-dir /models/BiRefNet",
    "hf download lokCX/4x-Ultrasharp 4x-UltraSharp.pth --local-dir /models/upscale"
)
if ($WithFlux) {
    if (-not $HfToken) { throw "FLUX klein is gated: set HF_TOKEN or pass -HfToken" }
    $steps += "hf download black-forest-labs/flux-3-image-klein --local-dir /models/black-forest-labs__flux-3-image-klein"
}
# Worker and API run as uid 10001; BiRefNet remote code is cached under /models/hf.
$steps += "mkdir -p /models/hf /models/loras && chown -R 10001:10001 /models"

$script = "set -e; pip install -q -U huggingface_hub; " + ($steps -join "; ")

docker run --rm `
  -v models_data:/models `
  -e HF_TOKEN=$HfToken `
  -e HF_HUB_ENABLE_HF_TRANSFER=0 `
  python:3.12-slim `
  bash -c $script

if ($LASTEXITCODE -ne 0) { throw "Model download failed (exit $LASTEXITCODE)" }
Write-Host "Models downloaded into Docker volume models_data"
