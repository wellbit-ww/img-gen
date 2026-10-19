from __future__ import annotations

import asyncio
import json
from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.schemas import JobResponse, JobStatus
from app.db.models import ImageRecord, JobRecord
from app.db.session import get_session
from app.services.events import job_channel, set_cancel_flag

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobResponse:
    job = session.get(JobRecord, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    images = session.exec(select(ImageRecord).where(ImageRecord.job_id == job_id)).all()
    return JobResponse(
        id=job.id,
        status=JobStatus(job.status),
        job_type=job.job_type,
        error=job.error,
        image_ids=[img.id for img in images],
        params=job.params,
    )


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    job = session.get(JobRecord, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    await set_cancel_flag(settings.redis_url, job_id)
    job.status = "cancelled"
    session.add(job)
    session.commit()
    return {"status": "cancelled"}


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    settings: Settings = Depends(get_app_settings),
) -> StreamingResponse:
    async def event_stream():
        client = redis.from_url(settings.redis_url)
        pubsub = client.pubsub()
        await pubsub.subscribe(job_channel(job_id))
        try:
            last = await client.get(f"job:{job_id}:last")
            if last:
                yield f"data: {last.decode() if isinstance(last, bytes) else last}\n\n"
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    yield f"data: {data}\n\n"
                    parsed: dict[str, Any] = json.loads(data)
                    if parsed.get("type") == "done":
                        break
                    if parsed.get("type") == "error":
                        break
                    if parsed.get("type") == "status" and parsed.get("status") == "cancelled":
                        break
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(job_channel(job_id))
            await client.aclose()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
