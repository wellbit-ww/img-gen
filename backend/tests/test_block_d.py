from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from sqlmodel import Session

from app.config import Settings
from app.db.models import JobRecord, LoraDataset, LoraTrainJob
from app.services.lora_dataset import LORA_LONG_SIDE, resize_dataset_image, write_caption
from app.services.lora_train import (
    build_ai_toolkit_yaml,
    install_trained_lora,
    wait_for_trainer,
    write_train_config,
)
from app.worker import tasks as worker_tasks
from app.worker.tasks import process_lora_train


def test_resize_dataset_image_long_side(tmp_path: Path):
    path = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000), color=(1, 2, 3)).save(path)
    resize_dataset_image(path)
    with Image.open(path) as img:
        assert max(img.size) == LORA_LONG_SIDE


def test_build_ai_toolkit_yaml_uses_qwen_image_2_arch(data_dir: Path):
    settings = Settings(engine_mode="mock", data_dir=data_dir, models_dir=data_dir / "models")
    yaml_doc = build_ai_toolkit_yaml(
        settings,
        "train-1",
        data_dir / "ds",
        rank=16,
        steps=500,
        learning_rate=1e-4,
        trigger_word="brandstyle",
    )
    proc = yaml_doc["config"]["process"][0]
    assert proc["model"]["arch"] == "qwen_image_2"
    assert 1328 in proc["datasets"][0]["resolution"]


def test_write_train_config_creates_file(data_dir: Path):
    settings = Settings(engine_mode="mock", data_dir=data_dir, models_dir=data_dir / "models")
    ds = data_dir / "lora_datasets" / "d1"
    ds.mkdir(parents=True)
    path = write_train_config(settings, "tid", ds, 32, 100, 1e-4, "tw")
    assert path.is_file()
    assert (data_dir / "lora_runs" / "tid" / "train.yaml").is_file()


@pytest.mark.asyncio
async def test_lora_train_mock_requires_real_mode(worker_db):
    job_id = "lora-job-d"
    train_id = "train-d-1"
    with Session(worker_tasks._engine_db) as session:
        session.add(
            LoraDataset(id="ds1", name="n", trigger_word="tw", path=str(worker_db.data_dir))
        )
        session.add(
            LoraTrainJob(
                id=train_id,
                dataset_id="ds1",
                config_path="",
                output_path=str(worker_db.data_dir / "lora_runs" / train_id),
            )
        )
        session.add(
            JobRecord(id=job_id, job_type="lora_train", status="queued", queue="quality")
        )
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
                    "steps": 50,
                    "learning_rate": 1e-4,
                    "trigger_word": "tw",
                }
            },
        )
    with Session(worker_tasks._engine_db) as session:
        job = session.get(JobRecord, job_id)
        assert job is not None
        assert job.status == "failed"
        assert "real" in (job.error or "").lower()


def test_wait_for_trainer_done_and_install(tmp_path: Path):
    settings = Settings(engine_mode="real", data_dir=tmp_path, models_dir=tmp_path / "models")
    run_dir = tmp_path / "lora_runs" / "t1"
    weights = run_dir / "workspace" / "t1"
    weights.mkdir(parents=True)
    (weights / "t1.safetensors").write_bytes(b"w")
    (run_dir / "done").touch()
    wait_for_trainer(run_dir, lambda: False, timeout_s=5, poll_s=0.01)
    dest = install_trained_lora(settings, "t1", run_dir)
    assert dest == tmp_path / "models" / "loras" / "t1" / "pytorch_lora_weights.safetensors"
    assert dest.read_bytes() == b"w"


def test_wait_for_trainer_failed_marker(tmp_path: Path):
    (tmp_path / "failed").touch()
    with pytest.raises(RuntimeError, match="ошибкой"):
        wait_for_trainer(tmp_path, lambda: False, timeout_s=5, poll_s=0.01)


def test_wait_for_trainer_not_picked_up(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.services.lora_train.TRAINER_PICKUP_TIMEOUT_S", 0.05)
    (tmp_path / "queued").touch()
    with pytest.raises(RuntimeError, match="Trainer не взял"):
        wait_for_trainer(tmp_path, lambda: False, timeout_s=5, poll_s=0.01)
    assert not (tmp_path / "queued").exists()


def test_update_caption_endpoint(client, data_dir: Path):
    ds_path = data_dir / "lora_datasets" / "cap-ds"
    ds_path.mkdir(parents=True)
    img = ds_path / "img_000.png"
    Image.new("RGB", (64, 64)).save(img)
    write_caption(img, "old")
    from app.db.models import LoraDataset
    from app.db.session import get_session

    gen = get_session()
    session = next(gen)
    try:
        session.add(
            LoraDataset(id="cap-ds", name="t", trigger_word="tw", path=str(ds_path))
        )
        session.commit()
    finally:
        gen.close()
    resp = client.put(
        "/api/lora/datasets/cap-ds/captions/img_000",
        data={"text": "new caption text"},
    )
    assert resp.status_code == 200
    assert (ds_path / "img_000.txt").read_text(encoding="utf-8") == "new caption text"
