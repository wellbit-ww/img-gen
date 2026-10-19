from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from app.config import Settings
from app.engines.base import Capability, StepCallback
from app.engines.bgremove import remove_background_birefnet
from app.engines.mock import MockEngine
from app.engines.upscaler import upscale_image


class UtilsEngine:
    name = "utils"
    capabilities: set[Capability] = {"upscale", "bgremove"}
    _mock: MockEngine

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._mock = MockEngine()

    def load(self) -> None:
        return

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        num_images: int,
        seed: int | None,
        negative_prompt: str,
        transparent: bool,
        on_step: StepCallback,
        cancel_check: Callable[[], bool],
        lora_id: str | None = None,
        lora_scale: float = 0.8,
    ) -> list[Image.Image]:
        if self._settings.engine_mode == "real":
            raise RuntimeError("Utils engine cannot generate images")
        return self._mock.generate(
            prompt,
            width,
            height,
            steps,
            guidance,
            num_images,
            seed,
            negative_prompt,
            transparent,
            on_step,
            cancel_check,
        )

    def edit(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image | None,
        references: list[Image.Image],
        strength: float,
        steps: int,
        seed: int | None,
        on_step: StepCallback,
        cancel_check: Callable[[], bool],
    ) -> list[Image.Image]:
        if self._settings.engine_mode == "real":
            raise RuntimeError("Utils engine cannot edit images")
        return self._mock.edit(
            prompt, image, mask, references, strength, steps, seed, on_step, cancel_check
        )

    def upscale(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        return upscale_image(
            image,
            target_w,
            target_h,
            self._settings.models_dir,
            engine_mode=self._settings.engine_mode,
        )

    def remove_background(self, image: Image.Image, method: str) -> Image.Image:
        if method == "qwen":
            raise RuntimeError(
                "Qwen background removal must run on the Qwen edit queue, not utils"
            )
        if method in ("birefnet", "auto"):
            return remove_background_birefnet(image, self._settings)
        if self._settings.engine_mode == "real":
            raise RuntimeError(f"Unsupported background removal method: {method}")
        return self._mock.remove_background(image, method)
