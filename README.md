# Presentation Image Generator (img-gen)

Локальный генератор изображений для презентаций: FastAPI + React, очередь Redis/arq, модели **Qwen-Image-2.1** на GPU-сервере, **mock-режим** на машине разработки без видеокарты.

## Быстрый старт (разработка, Windows)

1. Скопируйте `.env.example` → `.env` (`ENGINE_MODE=mock`).
2. Redis: `docker compose up -d redis` или локальный Redis/Memurai.
3. Бэкенд:

```powershell
cd backend
uv sync
uv run img-gen-api
```

4. Воркер (второй терминал):

```powershell
cd backend
uv run img-gen-worker
```

5. Фронтенд:

```powershell
cd frontend
npm install
npm run dev
```

Откройте http://localhost:5173 (прокси на API :8000).

## Тесты

```powershell
cd backend
uv run pytest
uv run ruff check .
```

## Продакшен

См. [docs/DEPLOY.md](docs/DEPLOY.md) — Docker на Windows 11 + 4× RTX 5090, `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`.

## Типичные проблемы

| Симптом | Решение |
|---------|---------|
| Задачи висят в `queued` | Запустите `img-gen-worker`, проверьте Redis |
| `degraded` в `/api/health` | Redis недоступен |
| OOM на 32 GB VRAM | Настройте стратегии в `config/workers.yaml`, используйте Turbo/draft |
| PyTorch на RTX 5090 | CUDA 12.8+ и wheels `cu128` |
| Кириллица на картинке | Режим **overlay** (текст через Pillow) |
| Медленная загрузка моделей | Именованный Docker-том `models_data`, не NTFS bind-mount |

## Структура

- `backend/app` — API, воркеры, движки (`engines/`), пресеты
- `config/presets` — YAML-стили для слайдов
- `frontend` — React UI
- `scripts/` — загрузка моделей, smoke-тест GPU
