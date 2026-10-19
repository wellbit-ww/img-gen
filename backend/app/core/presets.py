from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.core.aspect import resolve_size
from app.core.schemas import AspectRatio, GenerateRequest, PresetInfo, Quality, TextMode


class Preset:
    def __init__(self, data: dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.name: str = data.get("name", self.id)
        self.default_aspect = AspectRatio(data.get("default_aspect", "16:9"))
        self.transparent: bool = bool(data.get("transparent", False))
        self.text_mode = TextMode(data.get("text_mode", "none"))
        self.prompt_template: str = data.get("prompt_template", "{subject}")
        self.negative: str = data.get("negative", "")
        self.steps: dict[str, int] = data.get("steps", {"draft": 8, "standard": 30, "high": 50})
        self.guidance: float = float(data.get("guidance", 4.0))
        self.description: str = data.get("description", "")
        self.free_side: str = data.get("free_side", "left")
        self.lora_id: str | None = data.get("lora_id")

    def to_info(self) -> PresetInfo:
        return PresetInfo(
            id=self.id,
            name=self.name,
            default_aspect=self.default_aspect.value,
            transparent=self.transparent,
            text_mode=self.text_mode.value,
            description=self.description,
        )


class PresetStore:
    def __init__(self, presets_dir: Path) -> None:
        self._presets_dir = presets_dir
        self._cache: dict[str, Preset] = {}

    def reload(self) -> None:
        self._cache.clear()
        if not self._presets_dir.exists():
            return
        for path in sorted(self._presets_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if "id" not in data:
                data["id"] = path.stem
            self._cache[data["id"]] = Preset(data)

    def list(self) -> list[PresetInfo]:
        if not self._cache:
            self.reload()
        return [p.to_info() for p in self._cache.values()]

    def get(self, preset_id: str | None) -> Preset | None:
        if not self._cache:
            self.reload()
        if not preset_id:
            return None
        return self._cache.get(preset_id)


def build_generation_params(
    req: GenerateRequest,
    preset: Preset | None,
    brand_colors: list[str],
    subject_override: str | None = None,
) -> dict[str, Any]:
    subject = subject_override or req.prompt
    aspect = req.aspect_ratio
    quality = req.quality
    transparent = req.transparent
    text_mode = req.text_mode
    negative = req.negative_prompt
    steps = {"draft": 8, "standard": 30, "high": 50}[quality.value]
    guidance = 4.0
    lora_id = req.lora_id

    if preset:
        aspect = req.aspect_ratio or preset.default_aspect
        if req.transparent is False and preset.transparent:
            transparent = preset.transparent
        if req.text_mode == TextMode.NONE:
            text_mode = preset.text_mode
        negative = negative or preset.negative
        steps = preset.steps.get(quality.value, steps)
        guidance = preset.guidance
        palette = ", ".join(brand_colors) if brand_colors else ""
        prompt = preset.prompt_template.format(
            subject=subject,
            free_side=preset.free_side,
            palette=palette,
        )
        if not req.lora_id and preset.lora_id:
            lora_id = preset.lora_id
    else:
        prompt = subject

    if transparent:
        prefix = (
            "This is an RGBA image with transparency. "
            "The image has alpha channel and the background is transparent. "
        )
        prompt = prefix + prompt

    width, height = resolve_size(aspect, quality)
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance if quality != Quality.DRAFT else 1.0,
        "transparent": transparent,
        "text_mode": text_mode.value,
        "num_images": req.num_images,
        "seed": req.seed,
        "lora_id": lora_id,
        "lora_scale": req.lora_scale,
        "overlay": req.overlay.model_dump() if req.overlay else None,
        "preset_id": preset.id if preset else req.preset_id,
        "aspect_ratio": aspect.value,
        "quality": quality.value,
    }
