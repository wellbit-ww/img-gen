from __future__ import annotations

import os
from typing import Any

from app.config import Settings
from app.engines.base import ImageEngine
from app.engines.factory import create_engine

_engines: dict[str, ImageEngine] = {}


def get_cached_engine(settings: Settings, model_key: str) -> ImageEngine:
    if model_key not in _engines:
        engine = create_engine(settings, model_key)
        engine.load()
        _engines[model_key] = engine
    return _engines[model_key]


def engine_from_ctx(ctx: dict[str, Any], settings: Settings) -> ImageEngine:
    model_key = str(ctx.get("model_key") or os.environ.get("WORKER_MODEL", "mock"))
    if "engines" not in ctx:
        ctx["engines"] = {}
    engines: dict[str, ImageEngine] = ctx["engines"]
    if model_key not in engines:
        engines[model_key] = get_cached_engine(settings, model_key)
    return engines[model_key]
