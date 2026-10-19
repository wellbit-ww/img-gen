from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.presets import PresetStore
from app.core.schemas import ModelInfo, PresetInfo
from app.services.workers_config import load_workers_config

router = APIRouter(tags=["presets"])


@router.get("/presets", response_model=list[PresetInfo])
def list_presets(settings: Settings = Depends(get_app_settings)) -> list[PresetInfo]:
    path = settings.presets_dir
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return PresetStore(path).list()


@router.get("/models", response_model=list[ModelInfo])
def list_models(settings: Settings = Depends(get_app_settings)) -> list[ModelInfo]:
    models: dict[str, ModelInfo] = {}
    for worker in load_workers_config(settings):
        model_id = str(worker.get("model", "mock"))
        caps = ["t2i", "edit", "rgba"]
        if model_id == "utils":
            caps = ["upscale", "bgremove"]
        if model_id not in models:
            models[model_id] = ModelInfo(
                id=model_id,
                name=model_id,
                capabilities=caps,
                queues=list(worker.get("queues", [])),
                enabled=worker.get("enabled", True),
            )
        else:
            models[model_id].queues = list(set(models[model_id].queues + worker.get("queues", [])))
    return list(models.values())
