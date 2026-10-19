from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from app.config import Settings
from app.core.schemas import EnhancePromptRequest, EnhancePromptResponse, TextMode

CYRILLIC_IN_QUOTES = re.compile(r'"([^"]*[А-Яа-яЁё][^"]*)"')


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._system = (
            Path(__file__).resolve().parents[1] / "core" / "prompts" / "enhance.md"
        ).read_text(encoding="utf-8")

    async def enhance(self, req: EnhancePromptRequest) -> EnhancePromptResponse:
        user_payload = {
            "prompt": req.prompt,
            "preset_id": req.preset_id,
            "aspect_ratio": req.aspect_ratio.value,
            "text_mode": req.text_mode.value,
            "brand_colors": self._settings.brand_color_list,
        }
        schema = {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "negative_prompt": {"type": "string"},
                "overlay_title": {"type": "string"},
                "notes": {"type": "string"},
                "warning": {"type": "string"},
            },
            "required": ["prompt"],
        }
        body = {
            "model": self._settings.ollama_model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.4},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(f"{self._settings.ollama_url}/api/chat", json=body)
                resp.raise_for_status()
                data = resp.json()
                content = data["message"]["content"]
                parsed = json.loads(content)
        except Exception:
            warning = None
            if CYRILLIC_IN_QUOTES.search(req.prompt) and req.text_mode == TextMode.NATIVE:
                warning = "Для кириллицы в кавычках рекомендуется режим overlay."
            return EnhancePromptResponse(
                prompt=req.prompt,
                negative_prompt="",
                warning=warning,
                notes="Ollama unavailable; using original prompt.",
            )

        warning = parsed.get("warning")
        if CYRILLIC_IN_QUOTES.search(req.prompt) and req.text_mode == TextMode.NATIVE:
            warning = warning or "Для кириллицы в кавычках рекомендуется режим overlay."

        return EnhancePromptResponse(
            prompt=parsed.get("prompt", req.prompt),
            negative_prompt=parsed.get("negative_prompt", ""),
            overlay_title=parsed.get("overlay_title"),
            notes=parsed.get("notes"),
            warning=warning,
        )

    async def caption_image(self, image_path: Path, trigger_word: str) -> str:
        try:
            import base64

            b64 = base64.b64encode(image_path.read_bytes()).decode()
            body = {
                "model": self._settings.ollama_model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Describe this image for LoRA training. "
                            f"Start with trigger word '{trigger_word}'. One sentence."
                        ),
                        "images": [b64],
                    }
                ],
                "stream": False,
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._settings.ollama_url}/api/chat", json=body)
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
        except Exception:
            return f"{trigger_word}, corporate brand style illustration"
