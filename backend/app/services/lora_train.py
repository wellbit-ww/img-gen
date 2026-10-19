from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.config import Settings

TRAINER_TIMEOUT_S = 12 * 3600
TRAINER_PICKUP_TIMEOUT_S = 120


def lora_run_dir(settings: Settings, train_id: str) -> Path:
    return settings.data_dir / "lora_runs" / train_id


def resolve_qwen_model_path(settings: Settings) -> str:
    local = settings.models_dir / "Qwen__Qwen-Image-2.1"
    if local.is_dir():
        return str(local)
    alt = settings.models_dir / "Qwen__Qwen-Image-2.1-Turbo"
    if alt.is_dir():
        return str(alt)
    return "Qwen/Qwen-Image-2.1"


def build_ai_toolkit_yaml(
    settings: Settings,
    train_id: str,
    dataset_path: Path,
    rank: int,
    steps: int,
    learning_rate: float,
    trigger_word: str,
) -> dict[str, Any]:
    run_dir = lora_run_dir(settings, train_id)
    training_folder = str(run_dir / "workspace")
    return {
        "job": "extension",
        "config": {
            "name": train_id,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": training_folder,
                    "device": "cuda:0",
                    "trigger_word": trigger_word,
                    "network": {
                        "type": "lora",
                        "linear": rank,
                        "linear_alpha": rank,
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": max(50, steps // 8),
                        "max_step_saves_to_keep": 4,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset_path),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "resolution": [512, 768, 1024, 1328],
                            "cache_latents_to_disk": True,
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "cache_text_embeddings": True,
                        "steps": steps,
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "timestep_type": "shift",
                        "optimizer": "adamw8bit",
                        "lr": learning_rate,
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": resolve_qwen_model_path(settings),
                        "arch": "qwen_image_2",
                        "quantize": True,
                        "qtype": "convrot8",
                        "quantize_te": True,
                        "qtype_te": "convrot8",
                        "low_vram": True,
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": max(50, steps // 8),
                        "width": 1024,
                        "height": 1024,
                        "guidance_scale": 1.0,
                        "sample_steps": 30,
                        "prompts": [
                            f"{trigger_word}, presentation slide background, clean design",
                        ],
                    },
                }
            ],
        },
    }


def write_train_config(
    settings: Settings,
    train_id: str,
    dataset_path: Path,
    rank: int,
    steps: int,
    learning_rate: float,
    trigger_word: str,
) -> Path:
    run_dir = lora_run_dir(settings, train_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = build_ai_toolkit_yaml(
        settings, train_id, dataset_path, rank, steps, learning_rate, trigger_word
    )
    path = run_dir / "train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def unload_ollama_sync(settings: Settings) -> None:
    try:
        with httpx.Client(timeout=30.0) as client:
            client.post(
                f"{settings.ollama_url}/api/generate",
                json={"model": settings.ollama_model, "prompt": "", "keep_alive": 0},
            )
    except Exception:
        pass


def wait_for_trainer(
    run_dir: Path,
    cancel_check: Callable[[], bool],
    timeout_s: float = TRAINER_TIMEOUT_S,
    poll_s: float = 10.0,
) -> None:
    """Block until the trainer service marks the run done/failed (see train_entrypoint.sh)."""
    start = time.monotonic()
    started = False
    while time.monotonic() - start < timeout_s:
        if (run_dir / "done").exists():
            return
        if (run_dir / "failed").exists():
            raise RuntimeError(f"Обучение LoRA завершилось с ошибкой, см. {run_dir / 'train.log'}")
        if (run_dir / "running").exists():
            started = True
        if cancel_check():
            (run_dir / "queued").unlink(missing_ok=True)
            raise RuntimeError("Обучение LoRA отменено")
        if not started and time.monotonic() - start > TRAINER_PICKUP_TIMEOUT_S:
            (run_dir / "queued").unlink(missing_ok=True)
            raise RuntimeError(
                "Trainer не взял задачу: запустите "
                "`docker compose -f docker-compose.yml -f docker-compose.gpu.yml "
                "--profile training up -d trainer`"
            )
        time.sleep(poll_s)
    raise RuntimeError(f"Обучение LoRA не завершилось за {timeout_s / 3600:.0f} ч")


def install_trained_lora(settings: Settings, train_id: str, run_dir: Path) -> Path:
    """Copy the final ai-toolkit weights to models/loras/<id> where the engine loads them."""
    workspace = run_dir / "workspace" / train_id
    final = workspace / f"{train_id}.safetensors"
    if not final.is_file():
        candidates = sorted(workspace.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError(f"Trainer не сохранил веса LoRA в {workspace}")
        final = candidates[-1]
    dest_dir = settings.models_dir / "loras" / train_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "pytorch_lora_weights.safetensors"
    shutil.copy2(final, dest)
    return dest


def execute_lora_training(
    settings: Settings,
    train_id: str,
    job_id: str,
    payload: dict[str, Any],
    cancel_check: Callable[[], bool] = lambda: False,
) -> None:
    from sqlmodel import Session

    from app.db.models import LoraDataset, LoraTrainJob
    from app.worker.tasks import _engine_db

    if settings.engine_mode == "mock":
        raise RuntimeError(
            "Обучение LoRA доступно только при ENGINE_MODE=real и запущенном сервисе trainer"
        )

    rank = int(payload.get("rank", 32))
    steps = int(payload.get("steps", 2000))
    lr = float(payload.get("learning_rate", 1e-4))
    trigger_word = str(payload.get("trigger_word", "brandstyle"))
    dataset_path = Path(payload.get("dataset_path", ""))

    with Session(_engine_db) as session:
        train = session.get(LoraTrainJob, train_id)
        if not train:
            raise RuntimeError(f"LoRA train record {train_id} not found")
        dataset = session.get(LoraDataset, train.dataset_id)
        if dataset:
            dataset_path = Path(dataset.path)
            trigger_word = dataset.trigger_word
        train.status = "running"
        run_dir = lora_run_dir(settings, train_id)
        train.output_path = str(run_dir)
        session.add(train)
        session.commit()

    write_train_config(settings, train_id, dataset_path, rank, steps, lr, trigger_word)
    for marker in ("done", "failed", "running"):
        (run_dir / marker).unlink(missing_ok=True)
    unload_ollama_sync(settings)
    (run_dir / "queued").touch()
    log_path = run_dir / "train.log"
    try:
        wait_for_trainer(run_dir, cancel_check)
        install_trained_lora(settings, train_id, run_dir)
    except Exception as exc:
        with Session(_engine_db) as session:
            train = session.get(LoraTrainJob, train_id)
            if train:
                train.status = "failed"
                train.log = str(exc)
                session.add(train)
                session.commit()
        raise

    with Session(_engine_db) as session:
        train = session.get(LoraTrainJob, train_id)
        if train:
            train.status = "completed"
            if log_path.is_file():
                train.log = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            session.add(train)
            session.commit()
