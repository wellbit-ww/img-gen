# Deploy on Windows 11 + Docker (4× RTX 5090)

## Prerequisites

- NVIDIA driver 570+ with WSL2 GPU support (`nvidia-smi` inside WSL).
- Docker Desktop: WSL2 backend, **Use GPU** enabled.
- In `%UserProfile%\.wslconfig`: `memory=96GB` (or more), `swap=32GB`.
- Hugging Face token with access to gated models (optional FLUX).
- **64+ GB RAM** recommended when using CPU offload.

## Verify GPU in Docker

```powershell
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
```

## Configuration

1. Copy `.env.example` to `.env` and set `APP_API_KEY`, `HF_TOKEN`, `ENGINE_MODE=real`.
2. Download models into a **named Docker volume** (not slow NTFS bind-mount for weights):

```powershell
.\scripts\download_models.ps1
```

Or run downloads inside a one-off container mounting `models_data` volume.

## Start stack

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Pull Ollama model on the server:

```powershell
docker compose exec ollama ollama pull qwen3-vl:8b
```

## Smoke test

```powershell
docker compose exec worker uv run python /app/scripts/smoke_test_gpu.py
```

Tune `config/workers.yaml` memory strategies if OOM occurs.

## Training LoRA

```powershell
docker compose --profile training up -d trainer
```

Use the UI **Brand style** tab to upload images and start training; configs land in `/data/lora_configs`.

## Firewall

Allow TCP **8000** for LAN access, or put Caddy/IIS reverse proxy in front with HTTPS.

## Typical issues

| Issue | Fix |
|-------|-----|
| `no kernel image` on 5090 | Use CUDA 12.8+ PyTorch wheels (`cu128`) in GPU image |
| OOM on 32 GB | Enable `text_encoder_offload` or `fp8` in worker config |
| Slow model download | Use Docker named volume `models_data`, not `C:\` bind mount |
| SSE stuck | Ensure Redis and worker containers are running |
