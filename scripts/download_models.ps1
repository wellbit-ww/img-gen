param(
    [string]$HfToken = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"

docker volume create models_data | Out-Null

$models = @(
    @{ Repo = "Qwen/Qwen-Image-2.1"; Dir = "Qwen__Qwen-Image-2.1" },
    @{ Repo = "Qwen/Qwen-Image-2.1-Turbo"; Dir = "Qwen__Qwen-Image-2.1-Turbo" },
    @{ Repo = "ZhengPeng7/BiRefNet"; Dir = "BiRefNet" }
)

$flux = "black-forest-labs/flux-3-image-klein"

$hfCmd = @"
pip install -q 'huggingface_hub[cli]' && \
mkdir -p /models/upscale && \
"

foreach ($m in $models) {
    $hfCmd += "hf download $($m.Repo) --local-dir /models/$($m.Dir) && "
}

$hfCmd += @"
hf download alexoleg/4x-UltraSharp --include '4x-UltraSharp.pth' --local-dir /models/upscale && \
if [ -n \"\$HF_TOKEN\" ]; then hf download $flux --local-dir /models/flux-3-image-klein || true; fi
"@

docker run --rm `
  -v models_data:/models `
  -e HF_TOKEN=$HfToken `
  ghcr.io/astral-sh/uv:python3.12-bookworm `
  bash -c $hfCmd

Write-Host "Models downloaded into Docker volume models_data:/models"
