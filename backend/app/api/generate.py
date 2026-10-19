from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.presets import PresetStore, build_generation_params
from app.core.schemas import (
    BatchRequest,
    EditRequest,
    EnhancePromptRequest,
    EnhancePromptResponse,
    GenerateRequest,
    JobCreateResponse,
    RemoveBgRequest,
    TextMode,
    UpscaleRequest,
)
from app.db.models import ImageRecord
from app.db.session import get_session
from app.services.job_service import create_job
from app.services.ollama import OllamaClient

router = APIRouter(tags=["generation"])

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


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

    if req.text_mode == TextMode.NONE and CYRILLIC.search(req.prompt):
        req.text_mode = TextMode.OVERLAY

    params = build_generation_params(req, preset, settings.brand_color_list)
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
    params = build_generation_params(gen, preset, settings.brand_color_list)
    params["lines"] = [line.model_dump() for line in req.lines]
    job_id, queue, notice = await create_job(
        session,
        settings,
        "batch",
        params,
        quality=req.quality.value,
    )
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


@router.post("/edit", response_model=JobCreateResponse)
async def edit_image(
    req: EditRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> JobCreateResponse:
    if not req.image_id:
        raise HTTPException(400, "image_id required")
    image = session.get(ImageRecord, req.image_id)
    if not image:
        raise HTTPException(404, "Image not found")
    ref_paths = []
    for rid in req.reference_image_ids:
        ref = session.get(ImageRecord, rid)
        if ref:
            ref_paths.append(ref.path)
    params = {
        "prompt": req.prompt,
        "image_path": image.path,
        "reference_paths": ref_paths,
        "strength": req.strength,
        "steps": 30 if req.quality.value != "draft" else 8,
        "seed": req.seed,
    }
    job_id, queue, notice = await create_job(
        session, settings, "edit", params, quality=req.quality.value
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
