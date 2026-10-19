from __future__ import annotations

import json
from typing import Any

import redis

_clients: dict[str, redis.Redis] = {}


def get_sync_redis(redis_url: str) -> redis.Redis:
    if redis_url not in _clients:
        _clients[redis_url] = redis.from_url(redis_url, decode_responses=True)
    return _clients[redis_url]


def job_events_channel(job_id: str) -> str:
    return f"job:{job_id}:events"


def job_events_log_key(job_id: str) -> str:
    return f"job:{job_id}:log"


def publish_job_event(redis_url: str, job_id: str, event: dict[str, Any]) -> None:
    client = get_sync_redis(redis_url)
    payload = json.dumps(event, ensure_ascii=False)
    client.publish(job_events_channel(job_id), payload)
    client.rpush(job_events_log_key(job_id), payload)
    client.expire(job_events_log_key(job_id), 86400)
    client.set(f"job:{job_id}:last", payload, ex=86400)


def is_job_cancelled_sync(redis_url: str, job_id: str) -> bool:
    return bool(get_sync_redis(redis_url).exists(f"cancel:{job_id}"))


def worker_heartbeat_key(worker_id: str) -> str:
    return f"worker:hb:{worker_id}"


def set_worker_heartbeat(
    redis_url: str,
    worker_id: str,
    payload: dict[str, Any],
    ttl_seconds: int = 30,
) -> None:
    client = get_sync_redis(redis_url)
    client.setex(worker_heartbeat_key(worker_id), ttl_seconds, json.dumps(payload))


def list_worker_heartbeats(redis_url: str) -> list[dict[str, Any]]:
    client = get_sync_redis(redis_url)
    result: list[dict[str, Any]] = []
    for key in client.scan_iter(match="worker:hb:*"):
        raw = client.get(key)
        if raw:
            try:
                result.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return result


def batch_counter_key(parent_id: str, field: str) -> str:
    return f"batch:{parent_id}:{field}"
