from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session
from sqlmodel import create_engine as create_db_engine

from app.config import get_settings
from app.core.aspect import upscale_target_size
from app.core.schemas import OverlaySpec, TextMode
from app.core.text_overlay import apply_overlay
from app.db.models import JobRecord
from app.engines.base import JobCancelled
from app.engines.factory import create_engine
from app.services.events import is_cancelled, publish_event
from app.services.storage import create_image_record, output_path_for_job, save_image

_settings = get_settings()
_engine_db = create_db_engine(
    f"sqlite:///{_settings.db_path}",
    connect_args={"check_same_thread": False},
)


def _update_job(job_id: str, status: str, error: str | None = None) -> None:
    with Session(_engine_db) as session:
        job = session.get(JobRecord, job_id)
        if not job:
            return
        job.status = status
        job.error = error
        job.updated_at = datetime.now(UTC)
        session.add(job)
        session.commit()


def _publish(job_id: str, event: dict[str, Any]) -> None:
    asyncio.run(publish_event(_settings.redis_url, job_id, event))


def _cancel_check(job_id: str) -> bool:
    return asyncio.run(is_cancelled(_settings.redis_url, job_id))


def _get_engine(model_key: str):
    engine = create_engine(_settings, model_key)
    engine.load()
    return engine


def _save_images(
    job_id: str,
    images: list,
    params: dict[str, Any],
    prompt: str | None,
) -> list[str]:
    image_ids: list[str] = []
    overlay_data = params.get("overlay")
    text_mode = params.get("text_mode", "none")
    fonts_dir = _settings.fonts_dir
    if not fonts_dir.is_absolute():
        fonts_dir = (Path.cwd() / fonts_dir).resolve()

    with Session(_engine_db) as session:
        for idx, img in enumerate(images):
            path = output_path_for_job(_settings.data_dir, job_id, idx)
            meta = {"job_id": job_id, "params": params, "index": idx}
            save_image(img, path, meta)
            has_overlay = False
            final_path = path
            if text_mode == TextMode.OVERLAY.value and overlay_data:
                overlay = OverlaySpec(**overlay_data)
                if overlay.title or overlay.subtitle:
                    out = apply_overlay(img, overlay, fonts_dir)
                    final_path = path.with_name(path.stem + "_overlay.png")
                    save_image(out, final_path, {**meta, "overlay": overlay_data})
                    has_overlay = True
            record = create_image_record(
                job_id=job_id,
                path=final_path,
                params=params,
                prompt=prompt,
                preset_id=params.get("preset_id"),
                seed=params.get("seed"),
                has_overlay=has_overlay,
            )
            session.add(record)
            image_ids.append(record.id)
            _publish(job_id, {"type": "image", "image_id": record.id, "url": f"/api/images/{record.id}"})
        session.commit()
    return image_ids


async def process_generate(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    model_key = payload.get("model_key", "mock")
    params = payload["params"]
    _update_job(job_id, "running")
    _publish(job_id, {"type": "status", "status": "running"})

    def on_step(step: int, total: int) -> None:
        _publish(job_id, {"type": "progress", "step": step, "total": total})

    try:
        engine = _get_engine(model_key)
        images = engine.generate(
            prompt=params["prompt"],
            width=params["width"],
            height=params["height"],
            steps=params["steps"],
            guidance=params["guidance"],
            num_images=params["num_images"],
            seed=params.get("seed"),
            negative_prompt=params.get("negative_prompt", ""),
            transparent=params.get("transparent", False),
            on_step=on_step,
            cancel_check=lambda: _cancel_check(job_id),
            lora_id=params.get("lora_id"),
            lora_scale=params.get("lora_scale", 0.8),
        )
        ids = _save_images(job_id, images, params, params.get("prompt"))
        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "image_ids": ids})
    except JobCancelled:
        _update_job(job_id, "cancelled")
        _publish(job_id, {"type": "status", "status": "cancelled"})
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_edit(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from PIL import Image

    model_key = payload.get("model_key", "mock")
    params = payload["params"]
    _update_job(job_id, "running")

    def on_step(step: int, total: int) -> None:
        _publish(job_id, {"type": "progress", "step": step, "total": total})

    try:
        engine = _get_engine(model_key)
        base = Image.open(params["image_path"])
        mask = Image.open(params["mask_path"]) if params.get("mask_path") else None
        refs = [Image.open(p) for p in params.get("reference_paths", [])]
        images = engine.edit(
            prompt=params["prompt"],
            image=base,
            mask=mask,
            references=refs,
            strength=params.get("strength", 0.75),
            steps=params.get("steps", 30),
            seed=params.get("seed"),
            on_step=on_step,
            cancel_check=lambda: _cancel_check(job_id),
        )
        ids = _save_images(job_id, images, params, params.get("prompt"))
        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "image_ids": ids})
    except JobCancelled:
        _update_job(job_id, "cancelled")
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_upscale(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from PIL import Image

    params = payload["params"]
    _update_job(job_id, "running")
    try:
        engine = _get_engine("utils")
        img = Image.open(params["image_path"])
        tw, th = upscale_target_size(params.get("target", "4k"), img.width, img.height)
        out = engine.upscale(img, tw, th)
        ids = _save_images(job_id, [out], params, params.get("prompt"))
        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "image_ids": ids})
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_remove_bg(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from PIL import Image

    params = payload["params"]
    _update_job(job_id, "running")
    try:
        engine = _get_engine(payload.get("model_key", "utils"))
        img = Image.open(params["image_path"])
        out = engine.remove_background(img, params.get("method", "auto"))
        ids = _save_images(job_id, [out], params, params.get("prompt"))
        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "image_ids": ids})
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_batch(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    lines = payload["lines"]
    base_params = payload["params"]
    model_key = payload.get("model_key", "mock")
    all_ids: list[str] = []
    _update_job(job_id, "running")
    try:
        engine = _get_engine(model_key)
        for line_idx, line in enumerate(lines):
            params = dict(base_params)
            params["prompt"] = line["prompt"]
            params["num_images"] = base_params.get("num_images", 1)
            if line.get("overlay_title"):
                params.setdefault("overlay", {})
                params["overlay"]["title"] = line["overlay_title"]
                params["text_mode"] = TextMode.OVERLAY.value

            def on_step(step: int, total: int) -> None:
                _publish(
                    job_id,
                    {"type": "progress", "step": step, "total": total, "line": line_idx},
                )

            images = engine.generate(
                prompt=params["prompt"],
                width=params["width"],
                height=params["height"],
                steps=params["steps"],
                guidance=params["guidance"],
                num_images=params["num_images"],
                seed=params.get("seed"),
                negative_prompt=params.get("negative_prompt", ""),
                transparent=params.get("transparent", False),
                on_step=on_step,
                cancel_check=lambda: _cancel_check(job_id),
            )
            ids = _save_images(job_id, images, params, params["prompt"])
            all_ids.extend(ids)
        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "image_ids": all_ids})
    except JobCancelled:
        _update_job(job_id, "cancelled")
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_lora_train(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from app.db.models import LoraTrainJob

    _update_job(job_id, "running")
    with Session(_engine_db) as session:
        train = session.get(LoraTrainJob, job_id)
        if train:
            train.status = "running"
            session.add(train)
            session.commit()
    try:
        config_path = Path(payload["config_path"])
        out_dir = Path(payload["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        # Placeholder: real training invokes ai-toolkit in Docker profile.
        (out_dir / "README.txt").write_text(
            f"Training config at {config_path}. Run ai-toolkit container to execute.",
            encoding="utf-8",
        )
        with Session(_engine_db) as session:
            train = session.get(LoraTrainJob, job_id)
            if train:
                train.status = "completed"
                train.output_path = str(out_dir)
                session.add(train)
                session.commit()
        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "output": str(out_dir)})
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})
