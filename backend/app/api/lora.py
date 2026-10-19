from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.schemas import JobCreateResponse, LoraTrainRequest
from app.db.models import LoraDataset, LoraTrainJob
from app.db.session import get_session
from app.services.job_service import create_job
from app.services.ollama import OllamaClient

router = APIRouter(prefix="/lora", tags=["lora"])


@router.post("/datasets")
async def create_dataset(
    name: str = Form(...),
    trigger_word: str = Form("brandstyle"),
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
    files: list[UploadFile] = File(default=[]),
) -> dict[str, str]:
    dataset_id = str(uuid4())
    base = settings.data_dir / "lora_datasets" / dataset_id
    base.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(settings)
    for i, upload in enumerate(files):
        dest = base / f"img_{i:03d}.png"
        dest.write_bytes(await upload.read())
        caption = await client.caption_image(dest, trigger_word)
        (base / f"img_{i:03d}.txt").write_text(caption, encoding="utf-8")
    record = LoraDataset(
        id=dataset_id,
        name=name,
        trigger_word=trigger_word,
        path=str(base),
    )
    session.add(record)
    session.commit()
    return {"dataset_id": dataset_id}


@router.get("/datasets")
def list_datasets(session: Session = Depends(get_session)) -> list[dict]:
    rows = session.exec(select(LoraDataset)).all()
    return [
        {"id": r.id, "name": r.name, "trigger_word": r.trigger_word, "path": r.path}
        for r in rows
    ]


@router.get("")
def list_loras(settings: Settings = Depends(get_app_settings)) -> list[dict]:
    loras_dir = settings.models_dir / "loras"
    if not loras_dir.exists():
        return []
    return [{"id": p.name, "path": str(p)} for p in loras_dir.iterdir() if p.is_dir()]


@router.post("/train", response_model=JobCreateResponse)
async def train_lora(
    req: LoraTrainRequest,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> JobCreateResponse:
    dataset = session.get(LoraDataset, req.dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    train_id = str(uuid4())
    config_dir = settings.data_dir / "lora_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    out_dir = settings.models_dir / "loras" / train_id
    config = {
        "dataset_path": dataset.path,
        "trigger_word": dataset.trigger_word,
        "rank": req.rank,
        "steps": req.steps,
        "learning_rate": req.learning_rate,
        "output_dir": str(out_dir),
    }
    config_path = config_dir / f"{train_id}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    train = LoraTrainJob(
        id=train_id,
        dataset_id=dataset.id,
        config_path=str(config_path),
        output_path=str(out_dir),
    )
    session.add(train)
    session.commit()
    params = {
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "lines": [],
    }
    job_id, queue, notice = await create_job(
        session, settings, "lora_train", params, quality="high"
    )
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)
