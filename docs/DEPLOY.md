# Развёртывание на сервере (Windows 11 + Docker Desktop + 4× RTX 5090)

Команды для PowerShell, запускать из корня репозитория. Если не сказано иное, команда compose всегда такая:

```powershell
$dc = "docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
```

## 1. Подготовка сервера (один раз)

1. Драйвер NVIDIA 570+ (поддержка Blackwell / sm_120 и WSL2).
2. Docker Desktop: бэкенд WSL2, в Settings → Resources → WSL integration включена дистрибуция по умолчанию.
3. `%UserProfile%\.wslconfig` (после правки: `wsl --shutdown` и перезапуск Docker Desktop):

   ```ini
   [wsl2]
   memory=96GB
   swap=32GB
   ```

4. Проверка GPU в контейнере (должны быть видны 4 карты):

   ```powershell
   docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu24.04 nvidia-smi
   ```

5. Git и доступ к репозиторию: `winget install Git.Git`.

## 2. Перенос кода

```powershell
cd C:\srv
git clone -b cursor/presentation-image-generator https://github.com/wellbit-ww/img-gen.git
cd img-gen
```

Без доступа к GitHub: на машине разработки `git bundle create img-gen.bundle cursor/presentation-image-generator`, перенести файл, на сервере `git clone -b cursor/presentation-image-generator img-gen.bundle img-gen`.

Веса моделей, `node_modules`, `.venv` и `data/` переносить не нужно: образы собираются на сервере, фронтенд собирается внутри `Dockerfile.api`.

## 3. Настройка `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Обязательно:

| Переменная | Значение |
|------------|----------|
| `APP_API_KEY` | пароль для входа в UI (пустой = вход отключён) |
| `HF_TOKEN` | токен Hugging Face; нужен для FLUX klein (gated) и ускоряет загрузку |

`ENGINE_MODE`, `REDIS_URL`, `DATA_DIR`, `MODELS_DIR`, `OLLAMA_URL` задаются в compose-файлах и из `.env` не берутся.

## 4. Загрузка моделей в том `models_data`

```powershell
$env:HF_TOKEN = "hf_..."          # или оставить пустым без FLUX
.\scripts\download_models.ps1     # Qwen-Image-2.1, Turbo, BiRefNet, 4x-UltraSharp (~70+ ГБ)
.\scripts\download_models.ps1 -WithFlux   # опционально, FLUX klein (нужен принятый доступ на HF)
```

Скрипт кладёт веса в именованный том `models_data` (не на NTFS) и выставляет владельца uid 10001, от которого работают api и worker.

Проверка:

```powershell
docker run --rm -v models_data:/models alpine ls -la /models /models/upscale
```

## 5. Сборка и запуск

```powershell
Invoke-Expression "$dc build"
Invoke-Expression "$dc up -d"
Invoke-Expression "$dc exec ollama ollama pull qwen3-vl:8b"
Invoke-Expression "$dc ps"
```

Ожидаемо через 1–5 минут: `redis`, `api`, `worker`, `ollama` в статусе `healthy` / `running`.

```powershell
Invoke-RestMethod http://localhost:8000/api/health | ConvertTo-Json -Depth 4
```

В `workers` должны быть heartbeat-записи `gpu0-qwen-image-2.1`, `gpu1-qwen-image-2.1-turbo`, `gpu3-utils`; `engine_mode` = `real`.

UI: `http://<сервер>:8000`. Для доступа из сети откройте порт:

```powershell
New-NetFirewallRule -DisplayName "img-gen 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## 6. GPU smoke-тест

Smoke-тест сам загружает модели, поэтому на время теста воркер останавливают, чтобы не было OOM:

```powershell
Invoke-Expression "$dc stop worker"
Invoke-Expression "$dc run --rm worker python /scripts/smoke_test_gpu.py"
Invoke-Expression "$dc start worker"
```

При OOM поменяйте `memory:` у воркера в `config/workers.yaml` (`full` → `text_encoder_offload` → `fp8` → `model_cpu_offload`) и выполните `$dc restart worker`.

## 7. Обучение LoRA

1. Собрать образ trainer (долго, ставит ai-toolkit):

   ```powershell
   Invoke-Expression "$dc --profile training build trainer"
   ```

2. Запустить trainer как сервис. Он следит за `/data/lora_runs/*/queued`:

   ```powershell
   Invoke-Expression "$dc --profile training up -d trainer"
   ```

3. В UI или через API: `POST /api/lora/datasets` (15–40 картинок; ресайз до 1328 px, подписи через Ollama), правка подписей `PUT /api/lora/datasets/{id}/captions/{img_000}`, затем `POST /api/lora/train`.
4. Воркер пишет `/data/lora_runs/<id>/train.yaml` (ai-toolkit, `arch: qwen_image_2`), выгружает модель из Ollama (`keep_alive: 0`) и ставит маркер `queued`. Trainer обучает, пишет `train.log`, сэмплы в `workspace/` и маркер `done` или `failed`. Готовые веса копируются в `/models/loras/<id>/`.
5. Статус и лог: `GET /api/lora/train/<id>`.

Разовый ручной прогон одного конфига:

```powershell
Invoke-Expression "$dc --profile training run --rm trainer /data/lora_runs/<id>/train.yaml"
```

Trainer и utils-воркер оба на GPU 3. Пока идёт обучение, не запускайте апскейл и удаление фона.

## 8. Обновление

```powershell
git pull
Invoke-Expression "$dc up -d --build"
```

Миграции БД (`alembic upgrade head`) применяются автоматически при старте api и worker.

## 9. Логи и диагностика

```powershell
Invoke-Expression "$dc logs -f --tail 100 api worker"     # JSON (structlog)
Invoke-Expression "$dc exec worker nvidia-smi"
```

## Типичные проблемы

| Симптом | Причина и решение |
|---------|-------------------|
| `no kernel image is available` | старый PyTorch; в образе worker torch/torchvision `+cu128` из lock-файла — пересоберите `$dc build --no-cache worker` |
| worker `unhealthy` первые минуты | загрузка моделей; `start_period` 300 с. Смотрите `$dc logs worker` |
| `Permission denied` в `/models` | скрипт загрузки не сделал `chown`: `docker run --rm -v models_data:/models alpine chown -R 10001:10001 /models` |
| задачи висят в `queued` | воркер не слушает очередь: сверить `config/workers.yaml` и `/api/health` |
| OOM на 32 ГБ | стратегия `memory:` в `config/workers.yaml`, draft-качество / Turbo |
| «Улучшить» не меняет промпт, внизу «Ollama unavailable» | `$dc exec ollama ollama pull qwen3-vl:8b`, проверить `$dc ps ollama` |
| LoRA: «Trainer не взял задачу» | не запущен `$dc --profile training up -d trainer` |
| LoRA: `failed`, см. `train.log` | `docker run --rm -v app_data:/data alpine tail -50 /data/lora_runs/<id>/train.log` |
| в UI 401 | войдите с паролем `APP_API_KEY`; cookie живёт 7 дней |
| медленная загрузка весов | веса должны лежать в томе `models_data`, а не в bind-mount на `C:\` |
