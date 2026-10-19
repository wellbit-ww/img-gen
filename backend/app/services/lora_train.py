from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.config import Settings


def repo_root() -> Path:
    env = os.getenv("IMG_GEN_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[3]


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


def run_trainer_container(
    settings: Settings, train_id: str, _config_path: Path, log_path: Path
) -> None:
    if settings.engine_mode == "mock":
        raise RuntimeError(
            "Обучение LoRA доступно только при ENGINE_MODE=real и профиле Docker training"
        )
    compose = repo_root()
    compose_files = ["docker-compose.yml", "docker-compose.gpu.yml"]
    args = ["docker", "compose"]
    for f in compose_files:
        p = compose / f
        if p.is_file():
            args.extend(["-f", str(p)])
    args.extend(
        [
            "--profile",
            "training",
            "run",
            "--rm",
            "-e",
            f"TRAIN_CONFIG_PATH=/data/lora_runs/{train_id}/train.yaml",
            "trainer",
        ]
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Running: {' '.join(args)}\n")
        log.flush()
        proc = subprocess.run(
            args,
            cwd=str(compose),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"Trainer exited with code {proc.returncode}; see {log_path}")


def execute_lora_training(
    settings: Settings,
    train_id: str,
    job_id: str,
    payload: dict[str, Any],
) -> None:
    from sqlmodel import Session

    from app.db.models import LoraDataset, LoraTrainJob
    from app.worker.tasks import _engine_db

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

    yaml_path = write_train_config(
        settings, train_id, dataset_path, rank, steps, lr, trigger_word
    )
    unload_ollama_sync(settings)
    log_path = run_dir / "train.log"
    run_trainer_container(settings, train_id, yaml_path, log_path)

    with Session(_engine_db) as session:
        train = session.get(LoraTrainJob, train_id)
        if train:
            train.status = "completed"
            if log_path.is_file():
                train.log = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
            session.add(train)
            session.commit()
