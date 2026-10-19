from unittest.mock import patch

import pytest
from sqlmodel import Session
from sqlmodel import create_engine as create_db_engine

from app.config import get_settings
from app.db.models import JobRecord
from app.db.session import init_db
from app.worker import tasks


@pytest.mark.asyncio
async def test_process_generate_completes(data_dir):
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.db_path)
    tasks._settings = settings
    tasks._engine_db = create_db_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )

    job_id = "worker-test-job"
    with Session(tasks._engine_db) as session:
        session.add(JobRecord(id=job_id, job_type="generate", status="queued"))
        session.commit()

    params = {
        "prompt": "test slide",
        "width": 320,
        "height": 180,
        "steps": 2,
        "guidance": 1.0,
        "num_images": 1,
        "negative_prompt": "",
        "transparent": False,
    }
    with (
        patch("app.worker.tasks.publish_job_event"),
        patch("app.worker.tasks.is_job_cancelled_sync", return_value=False),
    ):
        await tasks.process_generate({}, job_id, {"model_key": "mock", "params": params})

    with Session(tasks._engine_db) as session:
        job = session.get(JobRecord, job_id)
        assert job is not None
        assert job.status == "completed"
