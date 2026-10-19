from __future__ import annotations

from app.config import Settings
from app.engines.base import ImageEngine
from app.engines.mock import MockEngine


def create_engine(settings: Settings, model_key: str) -> ImageEngine:
    if settings.engine_mode == "mock":
        return MockEngine()

    if model_key in ("qwen-image-2.1", "qwen-image-2.1-turbo"):
        from app.engines.qwen_image import QwenImageEngine

        return QwenImageEngine(settings=settings, variant=model_key)
    if model_key == "flux-alt":
        from app.engines.flux_alt import FluxAltEngine

        return FluxAltEngine(settings=settings)
    if model_key == "utils":
        from app.engines.utils_engine import UtilsEngine

        return UtilsEngine(settings=settings)

    return MockEngine()
