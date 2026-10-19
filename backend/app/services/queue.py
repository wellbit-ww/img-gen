from __future__ import annotations

from typing import Any

from arq import create_pool
from arq.connections import RedisSettings

from app.config import Settings


def redis_settings(settings: Settings) -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def enqueue_job(
    settings: Settings,
    function_name: str,
    job_id: str,
    payload: dict[str, Any],
    queue_name: str,
) -> None:
    redis = await create_pool(redis_settings(settings))
    try:
        await redis.enqueue_job(function_name, job_id, payload, _queue_name=queue_name)
    finally:
        await redis.close()


def pick_queue(job_type: str, quality: str, model: str = "auto") -> str:
    if job_type in ("upscale",):
        return "upscale"
    if job_type in ("remove_bg",):
        return "bgremove"
    if job_type == "edit":
        return "edit"
    if job_type == "batch":
        return "batch_parent"
    if job_type == "lora_train":
        return "lora"
    if quality == "draft":
        return "draft"
    if model == "alt":
        return "alt"
    return "quality"
