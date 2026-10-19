from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis


def job_channel(job_id: str) -> str:
    return f"job:{job_id}:events"


async def publish_event(redis_url: str, job_id: str, event: dict[str, Any]) -> None:
    client = redis.from_url(redis_url)
    try:
        await client.publish(job_channel(job_id), json.dumps(event, ensure_ascii=False))
        key = f"job:{job_id}:last"
        await client.set(key, json.dumps(event, ensure_ascii=False), ex=86400)
    finally:
        await client.aclose()


async def set_cancel_flag(redis_url: str, job_id: str) -> None:
    client = redis.from_url(redis_url)
    try:
        await client.set(f"cancel:{job_id}", "1", ex=3600)
    finally:
        await client.aclose()


async def is_cancelled(redis_url: str, job_id: str) -> bool:
    client = redis.from_url(redis_url)
    try:
        val = await client.get(f"cancel:{job_id}")
        return val is not None
    finally:
        await client.aclose()
