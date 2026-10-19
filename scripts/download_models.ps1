param(
    [string]$ModelsDir = ".\models",
    [string]$HfToken = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

docker run --rm -it `
  -v "${PWD}/models:/models" `
  -e HF_TOKEN=$HfToken `
  ghcr.io/astral-sh/uv:python3.12-bookworm `
  bash -c "pip install -q huggingface_hub && huggingface-cli download Qwen/Qwen-Image-2.1 --local-dir /models/Qwen__Qwen-Image-2.1 && huggingface-cli download Qwen/Qwen-Image-2.1-Turbo --local-dir /models/Qwen__Qwen-Image-2.1-Turbo"

Write-Host "Models downloaded to $ModelsDir"
