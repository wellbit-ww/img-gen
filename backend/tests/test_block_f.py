from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from app.config import get_settings
from app.core.presets import PresetStore, build_generation_params
from app.core.schemas import GenerateRequest
from app.db.models import ImageRecord, JobRecord
from app.db.session import get_session, init_db


def test_sse_replays_completed_job(client, data_dir):
    get_settings.cache_clear()
    settings = get_settings()
    init_db(settings.db_path)
    job_id = "sse-done-1"
    gen = get_session()
    session = next(gen)
    try:
        session.add(
            JobRecord(id=job_id, job_type="generate", status="completed", queue="draft")
        )
        session.add(
            ImageRecord(
                id="img-sse-1",
                job_id=job_id,
                path=str(data_dir / "x.png"),
            )
        )
        session.commit()
    finally:
        gen.close()
    with patch("app.api.jobs.fetch_job_events", return_value=[]):
        with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
            assert resp.status_code == 200
            lines = [ln for ln in resp.iter_lines() if ln]
    payload = json.loads(lines[0].removeprefix("data: "))
    assert payload.get("type") == "done"
    assert "img-sse-1" in payload.get("image_ids", [])


def test_icon_set_preset_resolution_2048():
    presets_dir = Path(__file__).resolve().parents[2] / "config" / "presets"
    store = PresetStore(presets_dir)
    preset = store.get("icon-set")
    req = GenerateRequest(prompt="icon", preset_id="icon-set")
    params = build_generation_params(req, preset, [])
    assert params["width"] == 2048
    assert params["height"] == 2048


def test_cancel_via_api(client):
    resp = client.post("/api/generate", json={"prompt": "cancel me", "num_images": 1})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    cancel = client.delete(f"/api/jobs/{job_id}")
    assert cancel.status_code == 200
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "cancelled"
