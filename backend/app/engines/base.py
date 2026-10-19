from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from PIL import Image

Capability = Literal["t2i", "edit", "inpaint", "rgba", "upscale", "bgremove"]

StepCallback = Callable[[int, int], None]


class JobCancelled(Exception):
    pass


class ImageEngine(Protocol):
    name: str
    capabilities: set[Capability]

    def load(self) -> None: ...

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
    ) -> list[Image.Image]: ...

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
    ) -> list[Image.Image]: ...

    def upscale(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image: ...

    def remove_background(self, image: Image.Image, method: str) -> Image.Image: ...
