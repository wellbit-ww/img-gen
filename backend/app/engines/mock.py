from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

from PIL import Image, ImageDraw, ImageFont

from app.engines.base import Capability, JobCancelled, StepCallback


class MockEngine:
    name = "mock"
    capabilities: set[Capability] = {"t2i", "edit", "inpaint", "rgba", "upscale", "bgremove"}

    def load(self) -> None:
        return

    def _placeholder(
        self,
        prompt: str,
        width: int,
        height: int,
        label: str,
        transparent: bool = False,
    ) -> Image.Image:
        mode = "RGBA" if transparent else "RGB"
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        hue = int(digest[:6], 16) % 360
        if transparent:
            img = Image.new(mode, (width, height), (40, 44, 52, 0))
        else:
            img = Image.new(mode, (width, height), (30 + hue % 40, 40, 60 + hue % 30))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width - 1, height - 1], outline=(255, 255, 255, 180), width=4)
        text = f"[MOCK {label}]\n{width}x{height}\n{prompt[:120]}"
        font = ImageFont.load_default()
        draw.multiline_text((24, 24), text, fill=(255, 255, 255, 255), font=font, spacing=6)
        return img

    def _simulate_steps(
        self,
        steps: int,
        on_step: StepCallback,
        cancel_check: Callable[[], bool],
    ) -> None:
        total = max(steps, 1)
        for i in range(1, total + 1):
            if cancel_check():
                raise JobCancelled()
            on_step(i, total)
            time.sleep(0.02)

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
        images: list[Image.Image] = []
        for n in range(num_images):
            self._simulate_steps(steps, on_step, cancel_check)
            images.append(
                self._placeholder(
                    f"{prompt}\nseed={seed} #{n+1} lora={lora_id}",
                    width,
                    height,
                    "T2I",
                    transparent=transparent,
                )
            )
        return images

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
        self._simulate_steps(steps, on_step, cancel_check)
        w, h = image.size
        return [self._placeholder(prompt, w, h, "EDIT")]

    def upscale(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    def remove_background(self, image: Image.Image, method: str) -> Image.Image:
        rgba = image.convert("RGBA")
        return self._placeholder(
            f"bgremove:{method}",
            rgba.width,
            rgba.height,
            "RGBA",
            transparent=True,
        )
