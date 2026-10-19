from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlmodel import Session, select

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.schemas import JobCreateResponse, LoraTrainRequest
from app.db.models import LoraDataset, LoraTrainJob
from app.db.session import get_session
from app.services.job_service import create_job
from app.services.lora_dataset import resize_dataset_image, write_caption
from app.services.lora_train import lora_run_dir
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
        suffix = Path(upload.filename or "img.png").suffix or ".png"
        dest = base / f"img_{i:03d}{suffix}"
        dest.write_bytes(await upload.read())
        resize_dataset_image(dest)
        caption = await client.caption_image(dest, trigger_word)
        write_caption(dest, caption)
    record = LoraDataset(
        id=dataset_id,
        name=name,
        trigger_word=trigger_word,
        path=str(base),
    )
    session.add(record)
    session.commit()
    return {"dataset_id": dataset_id}


@router.put("/datasets/{dataset_id}/captions/{image_stem}")
def update_caption(
    dataset_id: str,
    image_stem: str,
    text: str = Form(...),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    dataset = session.get(LoraDataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    base = Path(dataset.path)
    image_path = None
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = base / f"{image_stem}{ext}"
        if candidate.is_file():
            image_path = candidate
            break
    if image_path is None:
        raise HTTPException(404, "Image not found in dataset")
    write_caption(image_path, text)
    return {"status": "ok", "caption_path": str(image_path.with_suffix(".txt"))}


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
    run_dir = lora_run_dir(settings, train_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    train = LoraTrainJob(
        id=train_id,
        dataset_id=dataset.id,
        config_path=str(run_dir / "train.yaml"),
        output_path=str(run_dir),
    )
    session.add(train)
    session.commit()
    params = {
        "train_id": train_id,
        "dataset_path": dataset.path,
        "trigger_word": dataset.trigger_word,
        "rank": req.rank,
        "steps": req.steps,
        "learning_rate": req.learning_rate,
    }
    job_id, queue, notice = await create_job(
        session, settings, "lora_train", params, quality="high"
    )
    train.job_id = job_id
    session.add(train)
    session.commit()
    return JobCreateResponse(job_id=job_id, queue=queue, notice=notice)


@router.get("/train/{train_id}")
def get_lora_train(
    train_id: str,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> dict:
    train = session.get(LoraTrainJob, train_id)
    if not train:
        raise HTTPException(404, "Training job not found")
    out = Path(train.output_path or lora_run_dir(settings, train_id))
    samples = sorted({p.name for p in out.rglob("*.png")}) if out.is_dir() else []
    log_path = out / "train.log"
    log_text = train.log
    if log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "id": train.id,
        "job_id": train.job_id,
        "status": train.status,
        "log": log_text,
        "samples": samples,
        "run_dir": str(out),
    }
