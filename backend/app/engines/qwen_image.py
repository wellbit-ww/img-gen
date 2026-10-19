from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.engines.base import Capability, JobCancelled, StepCallback
from app.engines.memory import MemoryStrategy, apply_memory_strategy

RGBA_PREFIX = (
    "This is an RGBA image with transparency. "
    "The image has alpha channel and the background is transparent. "
)


class QwenImageEngine:
    name: str
    capabilities: set[Capability] = {"t2i", "edit", "inpaint", "rgba", "bgremove"}

    def __init__(self, settings: Settings, variant: str = "qwen-image-2.1") -> None:
        self._settings = settings
        self._variant = variant
        self.name = variant
        self._pipe = None
        self._memory: MemoryStrategy = "text_encoder_offload"
        self._lora_cache: dict[str, str] = {}

    def _repo_id(self) -> str:
        if self._variant == "qwen-image-2.1-turbo":
            return "Qwen/Qwen-Image-2.1-Turbo"
        return "Qwen/Qwen-Image-2.1"

    def _local_path(self) -> Path:
        return self._settings.models_dir / self._repo_id().replace("/", "__")

    def load(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        torch.backends.cuda.matmul.allow_tf32 = True
        path = self._local_path()
        source = str(path) if path.exists() else self._repo_id()
        self._pipe = QwenImage21Pipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            token=self._settings.hf_token,
        )
        try:
            self._pipe.to("cuda")
        except Exception:
            apply_memory_strategy(self._pipe, self._memory)

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            self.load()

    def _apply_lora(self, lora_id: str | None, scale: float) -> None:
        if not lora_id or self._pipe is None:
            return
        path = self._settings.models_dir / "loras" / lora_id
        if not path.exists():
            return
        self._pipe.load_lora_weights(str(path), adapter_name=lora_id)
        self._pipe.set_adapters([lora_id], adapter_weights=[scale])

    def _callback(
        self,
        on_step: StepCallback,
        cancel_check: Callable[[], bool],
    ):
        def _inner(_pipe, step: int, _timestep, _kwargs) -> None:
            if cancel_check():
                raise JobCancelled()
            total = getattr(_pipe, "_num_timesteps", None) or _kwargs.get("num_inference_steps", 1)
            on_step(step + 1, int(total))

        return _inner

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
        import torch

        self._ensure_loaded()
        self._apply_lora(lora_id, lora_scale)
        full_prompt = prompt
        if transparent and RGBA_PREFIX not in prompt:
            full_prompt = RGBA_PREFIX + prompt

        images: list[Image.Image] = []
        for i in range(num_images):
            if cancel_check():
                raise JobCancelled()
            gen_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            generator = torch.Generator(device="cuda").manual_seed(gen_seed + i)
            result = self._pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=self._callback(on_step, cancel_check),
            )
            images.extend(result.images)
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
        import torch

        self._ensure_loaded()
        gen_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        generator = torch.Generator(device="cuda").manual_seed(gen_seed)
        ref_input: Image.Image | list[Image.Image] = references[0] if len(references) == 1 else references
        if not references:
            ref_input = image
        kwargs: dict = {
            "prompt": prompt,
            "image": ref_input,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": self._callback(on_step, cancel_check),
        }
        if mask is not None:
            kwargs["mask_image"] = mask
        result = self._pipe(**kwargs)
        return list(result.images)

    def upscale(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        from app.engines.upscaler import upscale_image

        return upscale_image(image, target_w, target_h, self._settings.models_dir)

    def remove_background(self, image: Image.Image, method: str) -> Image.Image:
        if method == "birefnet":
            from app.engines.bgremove import remove_background_birefnet

            return remove_background_birefnet(image, self._settings)
        prompt = (
            RGBA_PREFIX
            + "Extract the main subject with clean edges on a fully transparent background."
        )
        return self.generate(
            prompt=prompt,
            width=image.width,
            height=image.height,
            steps=8,
            guidance=1.0,
            num_images=1,
            seed=42,
            negative_prompt="background, floor, shadow",
            transparent=True,
            on_step=lambda _s, _t: None,
            cancel_check=lambda: False,
        )[0]
