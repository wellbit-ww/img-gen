from __future__ import annotations

import redis
from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.schemas import HealthResponse
from app.services.redis_sync import list_worker_heartbeats
from app.services.workers_config import load_workers_config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    redis_ok = False
    try:
        client = redis.from_url(settings.redis_url)
        redis_ok = bool(client.ping())
        client.close()
    except Exception:
        redis_ok = False

    configured = load_workers_config(settings)
    heartbeats = list_worker_heartbeats(settings.redis_url) if redis_ok else []
    workers = heartbeats if heartbeats else configured
    return HealthResponse(
        status="ok" if redis_ok else "degraded",
        engine_mode=settings.engine_mode,
        redis=redis_ok,
        workers=workers,
    )
