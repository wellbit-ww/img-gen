from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlmodel import Session

from app.config import Settings
from app.db.models import JobRecord
from app.services.queue import enqueue_job, pick_queue
from app.services.workers_config import load_workers_config


def model_for_queue(settings: Settings, queue: str) -> str:
    for worker in load_workers_config(settings):
        if queue in worker.get("queues", []):
            return str(worker.get("model", "mock"))
    return "mock"


async def create_job(
    session: Session,
    settings: Settings,
    job_type: str,
    params: dict[str, Any],
    quality: str = "standard",
    model_choice: str = "auto",
) -> tuple[str, str, str | None]:
    requested_queue = pick_queue(job_type, quality, model_choice)
    queue = requested_queue
    notice: str | None = None
    configured_queues = set()
    for w in load_workers_config(settings):
        configured_queues.update(w.get("queues", []))
    if queue not in configured_queues:
        fallback = (
            "quality"
            if "quality" in configured_queues
            else next(iter(configured_queues), "draft")
        )
        notice = f"Queue '{queue}' is not configured; using '{fallback}'."
        queue = fallback

    job_id = str(uuid4())
    job = JobRecord(
        id=job_id,
        job_type=job_type,
        status="queued",
        queue=queue,
    )
    job.set_params(params)
    session.add(job)
    session.commit()

    model_key = model_for_queue(settings, queue)
    params_payload = dict(params)
    payload: dict[str, Any] = {"params": params_payload, "model_key": model_key}

    fn_map = {
        "generate": "process_generate",
        "edit": "process_edit",
        "upscale": "process_upscale",
        "remove_bg": "process_remove_bg",
        "batch": "process_batch",
        "lora_train": "process_lora_train",
        "batch_line": "process_batch_line",
    }
    fn = fn_map[job_type]
    if job_type == "batch":
        payload["lines"] = params_payload.pop("lines", [])
    await enqueue_job(settings, fn, job_id, payload, queue)
    return job_id, queue, notice
