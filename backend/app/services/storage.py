from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from PIL import Image

from app.core.schemas import OverlaySpec
from app.core.text_overlay import apply_overlay
from app.db.models import ImageRecord


def output_path_for_job(data_dir: Path, job_id: str, index: int, suffix: str = "png") -> Path:
    now = datetime.now(UTC)
    folder = data_dir / "outputs" / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{job_id}_{index}.{suffix}"


def save_image(
    image: Image.Image,
    path: Path,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode == "RGBA":
        image.save(path, format="PNG")
    else:
        image.save(path, format="PNG")
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def create_image_record(
    job_id: str,
    path: Path,
    params: dict,
    prompt: str | None = None,
    preset_id: str | None = None,
    seed: int | None = None,
    has_overlay: bool = False,
) -> ImageRecord:
    with Image.open(path) as img:
        w, h = img.size
    return ImageRecord(
        id=str(uuid4()),
        job_id=job_id,
        path=str(path),
        prompt=prompt,
        preset_id=preset_id,
        seed=seed,
        width=w,
        height=h,
        has_overlay=has_overlay,
        params_json=json.dumps(params, ensure_ascii=False),
    )


def apply_and_save_overlay(
    source_path: Path,
    overlay: OverlaySpec,
    fonts_dir: Path,
) -> Path:
    with Image.open(source_path) as img:
        out = apply_overlay(img, overlay, fonts_dir)
    out_path = source_path.with_name(source_path.stem + "_overlay.png")
    save_image(out, out_path, {"overlay": overlay.model_dump()})
    return out_path
