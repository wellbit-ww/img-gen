from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.config import Settings


def load_workers_config(settings: Settings) -> list[dict[str, Any]]:
    path = settings.workers_config
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        return [
            {
                "gpu": 0,
                "queues": ["quality", "draft", "edit", "batch", "upscale", "bgremove", "alt"],
                "model": "mock",
                "enabled": True,
            }
        ]
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    workers = data.get("workers", [])
    if settings.engine_mode == "mock":
        return [
            {
                "gpu": 0,
                "queues": ["quality", "draft", "edit", "batch", "upscale", "bgremove", "alt"],
                "model": "mock",
                "enabled": True,
            }
        ]
    return [w for w in workers if w.get("enabled", True)]
