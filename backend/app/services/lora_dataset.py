from __future__ import annotations

from pathlib import Path

from PIL import Image

LORA_LONG_SIDE = 1328


def resize_dataset_image(path: Path, long_side: int = LORA_LONG_SIDE) -> None:
    """Resize image so the longest side is at most long_side (preserves aspect)."""
    with Image.open(path) as img:
        w, h = img.size
        if max(w, h) <= long_side:
            return
        scale = long_side / max(w, h)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resized = img.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
        fmt = path.suffix.lower()
        if fmt in (".jpg", ".jpeg"):
            resized.save(path, format="JPEG", quality=92)
        else:
            resized.save(path, format="PNG")


def caption_path_for_image(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def write_caption(image_path: Path, text: str) -> Path:
    cap = caption_path_for_image(image_path)
    cap.write_text(text.strip(), encoding="utf-8")
    return cap
