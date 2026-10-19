from __future__ import annotations

from collections.abc import Callable

from PIL import Image

from app.config import Settings
from app.engines.base import Capability, JobCancelled, StepCallback
from app.engines.mock import MockEngine


class FluxAltEngine:
    """Optional FLUX Klein engine; falls back to mock when diffusers pipeline unavailable."""

    name = "flux-alt"
    capabilities: set[Capability] = {"t2i", "edit", "rgba"}
    _fallback: MockEngine

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipe = None
        self._fallback = MockEngine()

    def load(self) -> None:
        try:
            import torch
            from diffusers import Flux2KleinPipeline

            repo = "black-forest-labs/FLUX.2-klein-9B"
            local = self._settings.models_dir / repo.replace("/", "__")
            source = str(local) if local.exists() else repo
            self._pipe = Flux2KleinPipeline.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                token=self._settings.hf_token,
            )
            self._pipe.to("cuda")
        except Exception:
            self._pipe = None

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
        if self._pipe is None:
            return self._fallback.generate(
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
        import random

        import torch

        images: list[Image.Image] = []
        for i in range(num_images):
            if cancel_check():
                raise JobCancelled()
            gen_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            generator = torch.Generator(device="cuda").manual_seed(gen_seed + i)
            result = self._pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            images.extend(result.images)
            on_step(steps, steps)
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
        return self._fallback.edit(
            prompt, image, mask, references, strength, steps, seed, on_step, cancel_check
        )

    def upscale(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        return self._fallback.upscale(image, target_w, target_h)

    def remove_background(self, image: Image.Image, method: str) -> Image.Image:
        return self._fallback.remove_background(image, method)
