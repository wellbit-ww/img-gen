from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ENGINE_MODE", "mock")
os.environ.setdefault("APP_API_KEY", "")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    os.environ["DATA_DIR"] = str(d)
    presets = Path(__file__).resolve().parents[2] / "config" / "presets"
    os.environ["PRESETS_DIR"] = str(presets)
    return d


@pytest.fixture
def worker_db(data_dir: Path):
    from sqlmodel import create_engine

    import app.worker.tasks as worker_tasks
    from app.config import get_settings
    from app.db.session import init_db

    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.db_path)
    worker_tasks._settings = settings
    worker_tasks._engine_db = create_engine(
        f"sqlite:///{settings.db_path}",
        connect_args={"check_same_thread": False},
    )
    return settings


@pytest.fixture(autouse=True)
def mock_redis_enqueue(monkeypatch):
    async def _fake_enqueue(*_args, **_kwargs):
        return None

    monkeypatch.setattr("app.services.queue.enqueue_job", _fake_enqueue)
    async def _fake_cancel(*_a, **_k):
        return None

    monkeypatch.setattr("app.services.events.set_cancel_flag", _fake_cancel)


@pytest.fixture
def client(data_dir: Path) -> TestClient:
    from app.config import get_settings
    from app.db.session import init_db
    from app.main import create_app

    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.db_path)
    return TestClient(create_app())
