from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlmodel import create_engine as create_db_engine
from sqlmodel import select

import app.worker.tasks as worker_tasks
from app.config import get_settings
from app.db.models import ImageRecord, JobRecord
from app.db.session import get_session, init_db
from app.worker.tasks import process_generate


@pytest.mark.asyncio
async def test_mock_worker_writes_file_and_db(data_dir):
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.db_path)
    worker_tasks._settings = settings
    worker_tasks._engine_db = create_db_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )

    job_id = "integration-job-1"
    gen = get_session()
    session = next(gen)
    try:
        job = JobRecord(id=job_id, job_type="generate", status="queued", queue="draft")
        job.set_params(
            {
                "prompt": "Test slide",
                "width": 512,
                "height": 512,
                "steps": 4,
                "guidance": 1.0,
                "num_images": 1,
                "seed": 99,
                "negative_prompt": "",
                "transparent": False,
            }
        )
        session.add(job)
        session.commit()
        params = job.params
    finally:
        gen.close()

    ctx = {"model_key": "mock", "engines": {}}
    with (
        patch("app.worker.tasks.is_job_cancelled_sync", return_value=False),
        patch("app.worker.tasks.publish_job_event"),
    ):
        await process_generate(ctx, job_id, {"params": params, "model": "mock"})

    gen = get_session()
    session = next(gen)
    try:
        job = session.get(JobRecord, job_id)
        assert job is not None
        assert job.status == "completed"
        images = session.exec(select(ImageRecord).where(ImageRecord.job_id == job_id)).all()
        assert len(images) == 1
        assert images[0].seed == 99
        assert images[0].path.endswith(".png")
    finally:
        gen.close()


def test_num_images_max_eight(client):
    resp = client.post(
        "/api/generate",
        json={
            "prompt": "Too many",
            "num_images": 9,
        },
    )
    assert resp.status_code == 422
