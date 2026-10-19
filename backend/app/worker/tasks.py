from __future__ import annotations

import asyncio
import copy
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session
from sqlmodel import create_engine as create_db_engine

from app.config import get_settings
from app.core.aspect import upscale_target_size
from app.core.schemas import OverlaySpec, TextMode
from app.core.text_overlay import apply_overlay
from app.db.models import ImageRecord, JobRecord
from app.engines.base import JobCancelled
from app.services.redis_sync import (
    batch_counter_key,
    is_job_cancelled_sync,
    publish_job_event,
)
from app.services.storage import create_image_record, output_path_for_job, save_image
from app.worker.engine_cache import engine_from_ctx

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
    publish_job_event(_settings.redis_url, job_id, event)


def _cancel_check(job_id: str) -> bool:
    return is_job_cancelled_sync(_settings.redis_url, job_id)


def _resolve_seed(params: dict[str, Any]) -> int:
    seed = params.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    params["seed"] = seed
    return int(seed)


def _save_images(
    job_id: str,
    images: list,
    params: dict[str, Any],
    prompt: str | None,
    line_index: int | None = None,
) -> list[str]:
    image_ids: list[str] = []
    overlay_data = params.get("overlay")
    text_mode = params.get("text_mode", "none")
    fonts_dir = _settings.fonts_dir
    if not fonts_dir.is_absolute():
        fonts_dir = (Path.cwd() / fonts_dir).resolve()

    with Session(_engine_db) as session:
        for idx, img in enumerate(images):
            file_idx = (line_index or 0) * 100 + idx
            path = output_path_for_job(_settings.data_dir, job_id, file_idx)
            meta = {
                "job_id": job_id,
                "params": params,
                "index": file_idx,
                "seed": params.get("seed"),
            }
            save_image(img, path, meta)
            source_path = path
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
                source_path=str(source_path),
                params=params,
                prompt=prompt,
                preset_id=params.get("preset_id"),
                seed=params.get("seed"),
                has_overlay=has_overlay,
            )
            session.add(record)
            image_ids.append(record.id)
            _publish(
                job_id,
                {
                    "type": "image",
                    "image_id": record.id,
                    "url": f"/api/images/{record.id}",
                },
            )
        session.commit()
    return image_ids


def _generate_sync(
    ctx: dict[str, Any],
    job_id: str,
    params: dict[str, Any],
    model_key: str,
) -> list[str]:
    _resolve_seed(params)

    def on_step(step: int, total: int) -> None:
        _publish(job_id, {"type": "progress", "step": step, "total": total})

    engine = engine_from_ctx({**ctx, "model_key": model_key}, _settings)
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
    return _save_images(job_id, images, params, params.get("prompt"))


async def _run_guarded(
    job_id: str,
    fn,
    *args,
) -> None:
    _update_job(job_id, "running")
    _publish(job_id, {"type": "status", "status": "running"})
    try:
        result = await asyncio.to_thread(fn, *args)
        if isinstance(result, list):
            _update_job(job_id, "completed")
            _publish(job_id, {"type": "done", "image_ids": result})
        elif isinstance(result, dict):
            _update_job(job_id, "completed")
            _publish(job_id, {"type": "done", **result})
        else:
            _update_job(job_id, "completed")
            _publish(job_id, {"type": "done"})
    except JobCancelled:
        _update_job(job_id, "cancelled")
        _publish(job_id, {"type": "status", "status": "cancelled"})
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_generate(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    model_key = payload.get("model_key", "mock")
    params = payload["params"]
    await _run_guarded(job_id, _generate_sync, ctx, job_id, params, model_key)


async def process_edit(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from PIL import Image

    model_key = payload.get("model_key", "mock")
    params = payload["params"]
    _resolve_seed(params)

    def _edit_sync() -> list[str]:
        def on_step(step: int, total: int) -> None:
            _publish(job_id, {"type": "progress", "step": step, "total": total})

        engine = engine_from_ctx({**ctx, "model_key": model_key}, _settings)
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
        return _save_images(job_id, images, params, params.get("prompt"))

    await _run_guarded(job_id, _edit_sync)


async def process_upscale(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from PIL import Image

    params = payload["params"]

    def _upscale_sync() -> list[str]:
        engine = engine_from_ctx({**ctx, "model_key": "utils"}, _settings)
        img = Image.open(params["image_path"])
        tw, th = upscale_target_size(params.get("target", "4k"), img.width, img.height)
        out = engine.upscale(img, tw, th)
        return _save_images(job_id, [out], params, params.get("prompt"))

    await _run_guarded(job_id, _upscale_sync)


async def process_remove_bg(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from PIL import Image

    params = payload["params"]
    model_key = payload.get("model_key", "utils")

    def _bg_sync() -> list[str]:
        engine = engine_from_ctx({**ctx, "model_key": model_key}, _settings)
        img = Image.open(params["image_path"])
        out = engine.remove_background(img, params.get("method", "auto"))
        return _save_images(job_id, [out], params, params.get("prompt"))

    await _run_guarded(job_id, _bg_sync)


async def process_batch_line(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    """Generate a single batch line; job_id is the parent batch job."""
    parent_id = payload["parent_job_id"]
    line_index = int(payload["line_index"])
    params = copy.deepcopy(payload["params"])
    line = payload["line"]
    params["prompt"] = line["prompt"]
    params["num_images"] = payload.get("num_images", 1)
    if line.get("overlay_title"):
        overlay = copy.deepcopy(params.get("overlay") or {})
        overlay["title"] = line["overlay_title"]
        params["overlay"] = overlay
        params["text_mode"] = TextMode.OVERLAY.value

    model_key = payload.get("model_key", "mock")
    try:
        ids = await asyncio.to_thread(_generate_sync, ctx, parent_id, params, model_key)
        redis = _settings.redis_url
        from app.services.redis_sync import get_sync_redis

        client = get_sync_redis(redis)
        client.incr(batch_counter_key(parent_id, "done"))
        _publish(
            parent_id,
            {
                "type": "batch_line_done",
                "line": line_index,
                "image_ids": ids,
            },
        )
    except Exception as exc:
        from app.services.redis_sync import get_sync_redis

        client = get_sync_redis(_settings.redis_url)
        client.incr(batch_counter_key(parent_id, "failed"))
        _publish(parent_id, {"type": "batch_line_error", "line": line_index, "message": str(exc)})
        raise


async def process_batch(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    """Orchestrator: enqueue line jobs and wait for completion."""
    from app.services.queue import enqueue_job
    from app.services.redis_sync import get_sync_redis

    lines = payload["lines"]
    base_params = payload["params"]
    model_key = payload.get("model_key", "mock")
    num_images = base_params.get("num_images", 1)

    _update_job(job_id, "running")
    _publish(job_id, {"type": "status", "status": "running"})

    client = get_sync_redis(_settings.redis_url)
    client.set(batch_counter_key(job_id, "total"), len(lines))
    client.set(batch_counter_key(job_id, "done"), 0)
    client.set(batch_counter_key(job_id, "failed"), 0)

    try:
        for line_index, line in enumerate(lines):
            line_payload = {
                "parent_job_id": job_id,
                "line_index": line_index,
                "line": line,
                "params": copy.deepcopy(base_params),
                "model_key": model_key,
                "num_images": num_images,
            }
            await enqueue_job(
                _settings,
                "process_batch_line",
                f"{job_id}-line-{line_index}",
                line_payload,
                "batch",
            )

        total = len(lines)
        while True:
            if _cancel_check(job_id):
                raise JobCancelled()
            done = int(client.get(batch_counter_key(job_id, "done")) or 0)
            failed = int(client.get(batch_counter_key(job_id, "failed")) or 0)
            _publish(
                job_id,
                {
                    "type": "progress",
                    "step": done + failed,
                    "total": total,
                },
            )
            if done + failed >= total:
                break
            await asyncio.sleep(0.3)

        if failed:
            raise RuntimeError(f"Batch failed: {failed} line(s) errored")

        from sqlmodel import select

        with Session(_engine_db) as session:
            rows = session.exec(select(ImageRecord).where(ImageRecord.job_id == job_id)).all()
            all_ids = [r.id for r in rows]

        _update_job(job_id, "completed")
        _publish(job_id, {"type": "done", "image_ids": all_ids})
    except JobCancelled:
        _update_job(job_id, "cancelled")
        _publish(job_id, {"type": "status", "status": "cancelled"})
    except Exception as exc:
        _update_job(job_id, "failed", str(exc))
        _publish(job_id, {"type": "error", "message": str(exc)})


async def process_lora_train(ctx: dict, job_id: str, payload: dict[str, Any]) -> None:
    from app.db.models import LoraTrainJob

    train_id = payload.get("train_id", job_id)
    _update_job(job_id, "running")
    with Session(_engine_db) as session:
        train = session.get(LoraTrainJob, train_id)
        if train:
            train.status = "running"
            session.add(train)
            session.commit()
    try:
        config_path = Path(payload["config_path"])
        out_dir = Path(payload["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "README.txt").write_text(
            f"Training config at {config_path}. Run ai-toolkit container to execute.",
            encoding="utf-8",
        )
        with Session(_engine_db) as session:
            train = session.get(LoraTrainJob, train_id)
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
