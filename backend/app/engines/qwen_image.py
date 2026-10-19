from __future__ import annotations

import os
import random
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw

from app.config import Settings
from app.engines.base import Capability, JobCancelled, StepCallback
from app.engines.memory import MEMORY_FALLBACK_ORDER, MemoryStrategy, apply_memory_strategy

RGBA_PREFIX = (
    "This is an RGBA image with transparency. "
    "The image has alpha channel and the background is transparent. "
)

MAX_CONDITION_IMAGES = 10


def _apply_edit_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Paint edit region onto the condition image (pipeline has no separate mask arg)."""
    base = image.convert("RGB")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    m = mask.convert("L").resize(base.size)
    draw.bitmap((0, 0), m, fill=(255, 0, 0, 128))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def _condition_images(
    image: Image.Image,
    mask: Image.Image | None,
    references: list[Image.Image],
) -> list[Image.Image]:
    refs = list(references)[:MAX_CONDITION_IMAGES]
    if mask is not None:
        base = _apply_edit_mask(image, mask)
    else:
        base = image
    if not refs:
        return [base]
    remaining = MAX_CONDITION_IMAGES - 1
    return [base] + refs[:remaining]


class QwenImageEngine:
    name: str
    capabilities: set[Capability] = {"t2i", "edit", "inpaint", "rgba", "bgremove"}

    def __init__(self, settings: Settings, variant: str = "qwen-image-2.1") -> None:
        self._settings = settings
        self._variant = variant
        self.name = variant
        self._pipe = None
        self._memory: MemoryStrategy = os.environ.get(
            "WORKER_MEMORY", "text_encoder_offload"
        )  # type: ignore[assignment]
        self._loaded_lora: str | None = None

    def _repo_id(self) -> str:
        if self._variant == "qwen-image-2.1-turbo":
            return "Qwen/Qwen-Image-2.1-Turbo"
        return "Qwen/Qwen-Image-2.1"

    def _local_path(self) -> Path:
        return self._settings.models_dir / self._repo_id().replace("/", "__")

    def load(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        if self._settings.worker_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
        path = self._local_path()
        source = str(path) if path.exists() else self._repo_id()
        self._pipe = QwenImage21Pipeline.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            token=self._settings.hf_token,
        )
        if self._settings.worker_vae_tiling and hasattr(self._pipe, "vae"):
            self._pipe.vae.enable_tiling()
        apply_memory_strategy(self._pipe, self._memory)
        if self._settings.worker_torch_compile and hasattr(self._pipe, "transformer"):
            self._pipe.transformer = torch.compile(self._pipe.transformer)

    def _load_with_oom_fallback(self) -> None:
        if self._pipe is not None:
            return
        strategies = [self._memory]
        for s in MEMORY_FALLBACK_ORDER:
            if s not in strategies:
                strategies.append(s)
        last_err: Exception | None = None
        for strategy in strategies:
            self._memory = strategy
            try:
                self.load()
                return
            except Exception as exc:
                import torch

                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    last_err = exc
                    self._pipe = None
                    continue
                raise
        if last_err:
            raise RuntimeError(
                f"GPU out of memory after trying strategies: {strategies}"
            ) from last_err

    def _ensure_loaded(self) -> None:
        if self._pipe is None:
            self._load_with_oom_fallback()

    def _apply_lora(self, lora_id: str | None, scale: float) -> None:
        if not lora_id or self._pipe is None:
            return
        path = self._settings.models_dir / "loras" / lora_id
        if not path.exists():
            return
        if self._loaded_lora and self._loaded_lora != lora_id:
            self._pipe.unload_lora_weights()
            self._loaded_lora = None
        if self._loaded_lora != lora_id:
            self._pipe.load_lora_weights(str(path), adapter_name=lora_id)
            self._loaded_lora = lora_id
        self._pipe.set_adapters([lora_id], adapter_weights=[scale])

    @staticmethod
    def _true_cfg_scale(guidance: float, negative_prompt: str) -> float:
        """Qwen-Image-2.1 is sampled at true_cfg_scale=1.0 unless CFG is explicitly needed."""
        if guidance > 1.0 and negative_prompt.strip():
            return guidance
        return 1.0

    def _callback(
        self,
        on_step: StepCallback,
        cancel_check: Callable[[], bool],
        total_steps: int,
    ):
        def _inner(_pipe, step: int, _timestep, callback_kwargs: dict) -> dict:
            if cancel_check():
                raise JobCancelled()
            on_step(step + 1, total_steps)
            return callback_kwargs

        return _inner

    def _run_generate_once(
        self,
        full_prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        generator,
        on_step: StepCallback,
        cancel_check: Callable[[], bool],
    ):
        try:
            return self._pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=self._true_cfg_scale(guidance, negative_prompt),
                generator=generator,
                callback_on_step_end=self._callback(on_step, cancel_check, steps),
            )
        except Exception as exc:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                self._pipe = None
                self._load_with_oom_fallback()
                return self._pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt or None,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    true_cfg_scale=self._true_cfg_scale(guidance, negative_prompt),
                    generator=generator,
                    callback_on_step_end=self._callback(on_step, cancel_check, steps),
                )
            raise

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
            result = self._run_generate_once(
                full_prompt,
                negative_prompt,
                width,
                height,
                steps,
                guidance,
                generator,
                on_step,
                cancel_check,
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
        conditions = _condition_images(image, mask, references)
        img_arg: Image.Image | list[Image.Image] = (
            conditions[0] if len(conditions) == 1 else conditions
        )
        kwargs: dict = {
            "prompt": prompt,
            "image": img_arg,
            "num_inference_steps": steps,
            "generator": generator,
            "callback_on_step_end": self._callback(on_step, cancel_check, steps),
        }
        result = self._pipe(**kwargs)
        return list(result.images)

    def upscale(self, image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        from app.engines.upscaler import upscale_image

        return upscale_image(
            image,
            target_w,
            target_h,
            self._settings.models_dir,
            engine_mode=self._settings.engine_mode,
        )

    def remove_background(self, image: Image.Image, method: str) -> Image.Image:
        if method == "birefnet":
            from app.engines.bgremove import remove_background_birefnet

            return remove_background_birefnet(image, self._settings)
        prompt = (
            RGBA_PREFIX
            + "Extract the main subject with clean edges on a fully transparent background."
        )
        return self.edit(
            prompt=prompt,
            image=image,
            mask=None,
            references=[],
            strength=1.0,
            steps=8,
            seed=42,
            on_step=lambda _s, _t: None,
            cancel_check=lambda: False,
        )[0]
