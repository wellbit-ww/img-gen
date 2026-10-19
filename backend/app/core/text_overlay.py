from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.schemas import OverlaySpec


class FontNotFoundError(FileNotFoundError):
    pass


def _font_candidates(family: str) -> list[str]:
    return {
        "inter": ["Inter-Bold.ttf", "Inter-Regular.ttf"],
        "montserrat": ["Montserrat-Bold.ttf", "Montserrat-Regular.ttf"],
    }.get(family, ["Inter-Bold.ttf", "Inter-Regular.ttf"])


def _load_font(fonts_dir: Path, family: str, size: int) -> ImageFont.FreeTypeFont:
    for name in _font_candidates(family):
        path = fonts_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FontNotFoundError(
        f"No font files for '{family}' in {fonts_dir}. "
        "Add Inter/Montserrat TTF files with Cyrillic support."
    )


def _fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    family: str,
    fonts_dir: Path,
    max_width: int,
    start_size: int,
    min_size: int = 14,
) -> ImageFont.FreeTypeFont:
    size = start_size
    while size >= min_size:
        font = _load_font(fonts_dir, family, size)
        bbox = draw.textbbox((0, 0), text or "A", font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 2
    return _load_font(fonts_dir, family, min_size)


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
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

    title_font = _fit_font_size(
        draw,
        overlay.title,
        overlay.font,
        fonts_dir,
        max_text_width,
        max(24, base.width // 28),
    )
    subtitle_font = _fit_font_size(
        draw,
        overlay.subtitle,
        overlay.font,
        fonts_dir,
        max_text_width,
        max(16, base.width // 40),
    )

    title_lines = _wrap_text(draw, overlay.title, title_font, max_text_width)
    subtitle_lines = _wrap_text(draw, overlay.subtitle, subtitle_font, max_text_width)

    line_heights: list[int] = []
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        line_heights.append(int(bbox[3] - bbox[1]) + 8)
    for line in subtitle_lines:
        bbox = draw.textbbox((0, 0), line, font=subtitle_font)
        line_heights.append(int(bbox[3] - bbox[1]) + 6)

    block_h = sum(line_heights) + margin
    block_w = max_text_width + margin * 2

    if overlay.position in ("left", "top"):
        x0, y0 = margin, margin
    elif overlay.position == "center":
        x0, y0 = (base.width - block_w) // 2, (base.height - block_h) // 2
    elif overlay.position == "right":
        x0, y0 = base.width - block_w - margin, margin
    else:
        x0, y0 = margin, base.height - block_h - margin

    pad = margin // 2
    draw.rectangle(
        [x0 - pad, y0 - pad, x0 + block_w, y0 + block_h],
        fill=(0, 0, 0, 120),
    )

    y = y0
    for line in title_lines:
        draw.text(
            (x0, y),
            line,
            font=title_font,
            fill=overlay.color,
            stroke_width=1,
            stroke_fill="#000000",
        )
        bbox = draw.textbbox((x0, y), line, font=title_font)
        y += int(bbox[3] - bbox[1]) + 8
    for line in subtitle_lines:
        draw.text(
            (x0, y),
            line,
            font=subtitle_font,
            fill=overlay.color,
            stroke_width=1,
            stroke_fill="#000000",
        )
        bbox = draw.textbbox((x0, y), line, font=subtitle_font)
        y += int(bbox[3] - bbox[1]) + 6

    return base
