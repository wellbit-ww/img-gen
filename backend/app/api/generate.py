from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.presets import PresetStore, build_generation_params
from app.core.schemas import (
    BatchRequest,
    EnhancePromptRequest,
    EnhancePromptResponse,
    GenerateRequest,
    JobCreateResponse,
    Quality,
    RemoveBgRequest,
    TextMode,
    UpscaleRequest,
)
from app.db.models import ImageRecord
from app.db.session import get_session
from app.services.brand import get_brand_colors
from app.services.job_service import create_job
from app.services.ollama import OllamaClient

router = APIRouter(tags=["generation"])

CYRILLIC_IN_QUOTES = re.compile(r'"([^"]*[А-Яа-яЁё][^"]*)"')


def _preset_store(settings: Settings) -> PresetStore:
    path = settings.presets_dir
    if not path.is_absolute():
        from pathlib import Path

        path = (Path.cwd() / path).resolve()
    return PresetStore(path)


@router.post("/generate", response_model=JobCreateResponse)
async def generate(
    req: GenerateRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> JobCreateResponse:
    store = _preset_store(settings)
    preset = store.get(req.preset_id)
    if req.enhance_prompt:
        client = OllamaClient(settings)
        enhanced = await client.enhance(
            EnhancePromptRequest(
                prompt=req.prompt,
                preset_id=req.preset_id,
                aspect_ratio=req.aspect_ratio,
                text_mode=req.text_mode,
            )
        )
        req.prompt = enhanced.prompt
        if enhanced.negative_prompt:
            req.negative_prompt = enhanced.negative_prompt
        if enhanced.overlay_title and not req.overlay:
            from app.core.schemas import OverlaySpec

            req.overlay = OverlaySpec(title=enhanced.overlay_title)
            req.text_mode = TextMode.OVERLAY

    if req.text_mode is None and CYRILLIC_IN_QUOTES.search(req.prompt):
        req.text_mode = TextMode.OVERLAY

    brand_colors = get_brand_colors(session, settings)
    params = build_generation_params(req, preset, brand_colors)
    params["original_prompt"] = req.prompt
    job_id, queue, notice = await create_job(
        session,
        settings,
        "generate",
        params,
        quality=req.quality.value,
        model_choice=req.model.value,
    )
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


@router.post("/batch", response_model=JobCreateResponse)
async def batch_generate(
    req: BatchRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> JobCreateResponse:
    gen = GenerateRequest(
        prompt="batch",
        preset_id=req.preset_id,
        aspect_ratio=req.aspect_ratio,
        quality=req.quality,
        num_images=req.num_images,
        text_mode=req.text_mode,
    )
    store = _preset_store(settings)
    preset = store.get(req.preset_id)
    brand_colors = get_brand_colors(session, settings)
    params = build_generation_params(gen, preset, brand_colors)
    params["lines"] = [line.model_dump() for line in req.lines]
    job_id, queue, notice = await create_job(
        session,
        settings,
        "batch",
        params,
        quality=req.quality.value,
    )
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


async def _save_upload(
    settings: Settings, upload: UploadFile, suffix: str
) -> str:
    folder = settings.data_dir / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4()}{suffix}"
    dest.write_bytes(await upload.read())
    return str(dest)


@router.post("/edit", response_model=JobCreateResponse)
async def edit_image(
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
    prompt: str = Form(...),
    quality: Quality = Form(Quality.STANDARD),
    strength: float = Form(0.75),
    seed: int | None = Form(None),
    image_id: str | None = Form(None),
    reference_image_ids: str = Form(""),
    image: UploadFile | None = File(None),
    mask: UploadFile | None = File(None),
    references: list[UploadFile] = File(default=[]),
) -> JobCreateResponse:
    image_path: str | None = None
    if image_id:
        record = session.get(ImageRecord, image_id)
        if not record:
            raise HTTPException(404, "Image not found")
        image_path = record.path
    elif image is not None:
        suffix = Path(image.filename or "upload.png").suffix or ".png"
        image_path = await _save_upload(settings, image, suffix)
    else:
        raise HTTPException(400, "Provide image_id or image upload")

    ref_paths: list[str] = []
    for rid in [x.strip() for x in reference_image_ids.split(",") if x.strip()]:
        ref = session.get(ImageRecord, rid)
        if ref:
            ref_paths.append(ref.path)
    for ref_file in references[:10]:
        suffix = Path(ref_file.filename or "ref.png").suffix or ".png"
        ref_paths.append(await _save_upload(settings, ref_file, suffix))
    ref_paths = ref_paths[:10]

    mask_path: str | None = None
    if mask is not None:
        mask_path = await _save_upload(settings, mask, ".png")

    params = {
        "prompt": prompt,
        "image_path": image_path,
        "mask_path": mask_path,
        "reference_paths": ref_paths,
        "strength": strength,
        "steps": 30 if quality.value != "draft" else 8,
        "seed": seed,
    }
    job_id, queue, notice = await create_job(
        session, settings, "edit", params, quality=quality.value
    )
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


@router.post("/upscale", response_model=JobCreateResponse)
async def upscale(
    req: UpscaleRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> JobCreateResponse:
    image = session.get(ImageRecord, req.image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    params = {
        "image_path": image.path,
        "target": req.target,
        "prompt": image.prompt,
    }
    job_id, queue, notice = await create_job(session, settings, "upscale", params, quality="high")
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


@router.post("/remove-bg", response_model=JobCreateResponse)
async def remove_bg(
    req: RemoveBgRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> JobCreateResponse:
    image = session.get(ImageRecord, req.image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    params = {
        "image_path": image.path,
        "method": req.method,
        "prompt": image.prompt,
    }
    job_id, queue, notice = await create_job(
        session, settings, "remove_bg", params, quality="standard"
    )
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


@router.post("/enhance-prompt", response_model=EnhancePromptResponse)
async def enhance_prompt(
    req: EnhancePromptRequest,
    settings: Settings = Depends(get_app_settings),
) -> EnhancePromptResponse:
    client = OllamaClient(settings)
    return await client.enhance(req)
