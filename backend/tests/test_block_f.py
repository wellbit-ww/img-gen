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


def test_overlay_keeps_opaque_image_opaque():
    from PIL import Image

    from app.core.schemas import OverlaySpec
    from app.core.text_overlay import apply_overlay

    fonts = Path(__file__).resolve().parents[2] / "assets" / "fonts"
    img = Image.new("RGB", (1376, 768), (30, 40, 60))
    out = apply_overlay(img, OverlaySpec(title="Облачная платформа 2027"), fonts)
    assert out.getchannel("A").getextrema() == (255, 255)


def test_export_is_attachment(client, data_dir):
    from PIL import Image

    path = data_dir / "exp.png"
    Image.new("RGB", (64, 64)).save(path)
    gen = get_session()
    session = next(gen)
    try:
        session.add(ImageRecord(id="img-exp-1", job_id="j", path=str(path)))
        session.commit()
    finally:
        gen.close()
    resp = client.get("/api/images/img-exp-1/export?format=jpeg")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.headers["content-disposition"] == 'attachment; filename="img-exp-1.jpg"'


def test_login_cookie_protects_api(data_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app

    monkeypatch.setenv("APP_API_KEY", "s3cret")
    get_settings.cache_clear()
    try:
        init_db(get_settings().db_path)
        c = TestClient(create_app())
        assert c.get("/api/presets").status_code == 401
        assert c.get("/api/health").status_code == 200
        assert c.post("/api/login", data={"password": "wrong"}).status_code == 401
        ok = c.post("/api/login", data={"password": "s3cret"})
        assert ok.status_code == 200
        assert "img_gen_session" in ok.cookies
        assert c.get("/api/presets").status_code == 200
    finally:
        monkeypatch.setenv("APP_API_KEY", "")
        get_settings.cache_clear()


def test_cancel_via_api(client):
    resp = client.post("/api/generate", json={"prompt": "cancel me", "num_images": 1})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    cancel = client.delete(f"/api/jobs/{job_id}")
    assert cancel.status_code == 200
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "cancelled"
