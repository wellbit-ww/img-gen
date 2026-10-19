from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine, select

from app.config import get_settings
from app.db.models import ImageRecord, JobRecord
from app.db.session import init_db
from app.services.redis_sync import set_worker_heartbeat, worker_heartbeat_key
from app.worker import tasks as worker_tasks
from app.worker.gpu_lock import gpu_job_lock
from app.worker.tasks import (
    _should_abort_job,
    _update_job,
    process_batch,
    process_batch_line,
    process_generate,
    process_lora_train,
)
from tests.fake_redis import FakeRedis


@pytest.fixture
def worker_db(data_dir: Path):
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.db_path)
    worker_tasks._settings = settings
    worker_tasks._engine_db = create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )
    return settings


@pytest.mark.asyncio
async def test_batch_three_lines_overlay_zip(worker_db, data_dir):
    fake = FakeRedis()
    parent_id = "batch-parent-1"
    ctx = {"model_key": "mock", "engines": {}}

    with Session(worker_tasks._engine_db) as session:
        job = JobRecord(id=parent_id, job_type="batch", status="queued", queue="batch_parent")
        job.set_params({"num_images": 1, "width": 512, "height": 512, "steps": 4, "guidance": 1.0})
        session.add(job)
        session.commit()

    lines = [
        {"prompt": "line one", "overlay_title": "Title one"},
        {"prompt": "line two", "overlay_title": "Title two"},
        {"prompt": "line three", "overlay_title": "Title three"},
    ]
    payload = {
        "lines": lines,
        "params": {
            "width": 512,
            "height": 512,
            "steps": 4,
            "guidance": 1.0,
            "num_images": 1,
            "negative_prompt": "",
            "transparent": False,
            "text_mode": "none",
            "overlay": {"title": "", "subtitle": "", "position": "bottom"},
        },
        "model_key": "mock",
    }

    async def inline_enqueue(_settings, fn_name, line_job_id, line_payload, queue_name):
        assert queue_name == "batch"
        await process_batch_line(ctx, line_job_id, line_payload)

    with (
        patch("app.services.redis_sync.get_sync_redis", return_value=fake),
        patch("app.worker.tasks.publish_job_event"),
        patch("app.services.redis_sync.publish_job_event"),
        patch("app.worker.tasks.is_job_cancelled_sync", return_value=False),
        patch("app.services.queue.enqueue_job", side_effect=inline_enqueue),
    ):
        await process_batch(ctx, parent_id, payload)

    with Session(worker_tasks._engine_db) as session:
        job = session.get(JobRecord, parent_id)
        assert job is not None
        assert job.status == "completed"
        images = session.exec(select(ImageRecord).where(ImageRecord.job_id == parent_id)).all()
        assert len(images) == 3
        for img in images:
            assert img.has_overlay is True

    from app.main import create_app

    client = TestClient(create_app())
    resp = client.get(f"/api/jobs/{parent_id}/download.zip")
    assert resp.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    assert len(zf.namelist()) == 3


def test_gpu_lock_serializes_parallel_work():
    events: list[tuple[str, str]] = []

    def work(name: str) -> None:
        with gpu_job_lock():
            events.append((name, "start"))
            time.sleep(0.05)
            events.append((name, "end"))

    t1 = threading.Thread(target=work, args=("a",))
    t2 = threading.Thread(target=work, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert events.index(("a", "start")) < events.index(("a", "end"))
    assert events.index(("b", "start")) < events.index(("b", "end"))
    for i, ev in enumerate(events):
        if ev[1] == "start" and i + 1 < len(events):
            assert events[i + 1][1] == "end"


def test_heartbeat_ttl_refresh(worker_db):
    fake = FakeRedis()
    settings = worker_db
    with patch("app.services.redis_sync.get_sync_redis", return_value=fake):
        set_worker_heartbeat(settings.redis_url, "w1", {"worker_id": "w1", "busy": False})
        key = worker_heartbeat_key("w1")
        assert key in fake.store
        set_worker_heartbeat(settings.redis_url, "w1", {"worker_id": "w1", "busy": False})
        assert key in fake.store


def test_alembic_upgrade_on_empty_db(tmp_path: Path):
    db_path = (tmp_path / "nested" / "app.db").resolve()
    init_db(db_path)
    assert db_path.exists()
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    with Session(engine) as session:
        session.exec(select(JobRecord)).all()


@pytest.mark.asyncio
async def test_cancel_before_start(worker_db):
    fake = FakeRedis()
    fake.set("cancel:job-cancel-1", "1")
    job_id = "job-cancel-1"
    with Session(worker_tasks._engine_db) as session:
        session.add(JobRecord(id=job_id, job_type="generate", status="queued", queue="draft"))
        session.commit()
    with (
        patch("app.worker.tasks.is_job_cancelled_sync", return_value=True),
        patch("app.worker.tasks.publish_job_event"),
    ):
        assert _should_abort_job(job_id) is True
        payload = {"params": {}, "model_key": "mock"}
        await process_generate({"model_key": "mock", "engines": {}}, job_id, payload)
    with Session(worker_tasks._engine_db) as session:
        job = session.get(JobRecord, job_id)
        assert job is not None
        assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_during_run_not_overwritten(worker_db):
    job_id = "job-cancel-2"
    with Session(worker_tasks._engine_db) as session:
        session.add(JobRecord(id=job_id, job_type="generate", status="cancelled", queue="draft"))
        session.commit()
    _update_job(job_id, "completed")
    with Session(worker_tasks._engine_db) as session:
        job = session.get(JobRecord, job_id)
        assert job is not None
        assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_lora_train_fails_clearly(worker_db):
    from app.db.models import LoraDataset, LoraTrainJob

    job_id = "lora-job-1"
    train_id = "train-a-1"
    with Session(worker_tasks._engine_db) as session:
        session.add(
            LoraDataset(id="ds-a", name="d", trigger_word="tw", path=str(worker_db.data_dir))
        )
        session.add(
            LoraTrainJob(
                id=train_id,
                dataset_id="ds-a",
                config_path="",
                output_path=str(worker_db.data_dir / "lora_runs" / train_id),
            )
        )
        session.add(JobRecord(id=job_id, job_type="lora_train", status="queued", queue="quality"))
        session.commit()
    with (
        patch("app.worker.tasks.publish_job_event"),
        patch("app.worker.tasks.is_job_cancelled_sync", return_value=False),
        patch("app.services.lora_train.unload_ollama_sync"),
    ):
        await process_lora_train(
            {},
            job_id,
            {
                "params": {
                    "train_id": train_id,
                    "dataset_path": str(worker_db.data_dir),
                    "rank": 16,
                    "steps": 10,
                    "learning_rate": 1e-4,
                }
            },
        )
    with Session(worker_tasks._engine_db) as session:
        job = session.get(JobRecord, job_id)
        assert job is not None
        assert job.status == "failed"
        err = (job.error or "").lower()
        assert "real" in err or "lora" in err or "trainer" in err
