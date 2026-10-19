from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.schemas import OverlaySpec

RGBA_TRANSPARENT = (0, 0, 0, 0)


def _load_font(fonts_dir: Path, family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = {
        "inter": ["Inter-Bold.ttf", "Inter-Regular.ttf", "inter.ttf"],
        "montserrat": ["Montserrat-Bold.ttf", "Montserrat-Regular.ttf", "montserrat.ttf"],
    }
    for name in candidates.get(family, candidates["inter"]):
        path = fonts_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text.strip():
        return []
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def apply_overlay(image: Image.Image, overlay: OverlaySpec, fonts_dir: Path) -> Image.Image:
    base = image.convert("RGBA")
    draw = ImageDraw.Draw(base)
    margin = int(min(base.width, base.height) * 0.05)
    max_text_width = int(base.width * 0.45)

    title_font = _load_font(fonts_dir, overlay.font, max(24, base.width // 28))
    subtitle_font = _load_font(fonts_dir, overlay.font, max(16, base.width // 40))

    title_lines = _wrap_text(draw, overlay.title, title_font, max_text_width)
    subtitle_lines = _wrap_text(draw, overlay.subtitle, subtitle_font, max_text_width)

    line_heights: list[int] = []
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_heights.append(bbox[3] - bbox[1] + 8)
    for line in subtitle_lines:
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        line_heights.append(bbox[3] - bbox[1] + 6)

    block_h = sum(line_heights) + margin
    block_w = max_text_width + margin * 2

    if overlay.position in ("left", "top"):
        x0, y0 = margin, margin
    elif overlay.position == "center":
        x0, y0 = (base.width - block_w) // 2, (base.height - block_h) // 2
    elif overlay.position == "right":
        x0, y0 = base.width - block_w - margin, margin
    else:  # bottom
        x0, y0 = margin, base.height - block_h - margin

    pad = margin // 2
    draw.rectangle(
        [x0 - pad, y0 - pad, x0 + block_w, y0 + block_h],
        fill=(0, 0, 0, 120),
    )

    y = y0
    for line in title_lines:
        draw.text((x0, y), line, font=title_font, fill=overlay.color)
        bbox = draw.textbbox((x0, y), line, font=title_font)
        y += bbox[3] - bbox[1] + 8
    for line in subtitle_lines:
        draw.text((x0, y), line, font=subtitle_font, fill=overlay.color)
        bbox = draw.textbbox((x0, y), line, font=subtitle_font)
        y += bbox[3] - bbox[1] + 6

    return base
