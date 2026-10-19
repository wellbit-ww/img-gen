from __future__ import annotations

import json
from typing import Any

import redis.asyncio as redis

from app.services.redis_sync import (
    get_sync_redis,
    job_events_channel,
    job_events_log_key,
    publish_job_event,
)


def job_channel(job_id: str) -> str:
    return job_events_channel(job_id)


async def publish_event(redis_url: str, job_id: str, event: dict[str, Any]) -> None:
    publish_job_event(redis_url, job_id, event)


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


async def fetch_job_events(redis_url: str, job_id: str) -> list[dict[str, Any]]:
    client = redis.from_url(redis_url, decode_responses=True)
    try:
        raw_items = await client.lrange(job_events_log_key(job_id), 0, -1)
        events: list[dict[str, Any]] = []
        for item in raw_items:
            try:
                events.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return events
    finally:
        await client.aclose()


def fetch_job_events_sync(redis_url: str, job_id: str) -> list[dict[str, Any]]:
    client = get_sync_redis(redis_url)
    events: list[dict[str, Any]] = []
    for item in client.lrange(job_events_log_key(job_id), 0, -1):
        try:
            events.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    return events
