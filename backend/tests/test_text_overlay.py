from pathlib import Path

from PIL import Image

from app.core.schemas import OverlaySpec
from app.core.text_overlay import apply_overlay

FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def test_overlay_changes_pixels(tmp_path: Path):
    img = Image.new("RGB", (400, 300), (30, 60, 90))
    out = apply_overlay(
        img,
        OverlaySpec(title="Тест", subtitle="Подзаголовок", color="#FFFFFF"),
        FONTS_DIR,
    )
    assert out.mode == "RGBA"
    assert out.getpixel((10, 10)) != img.getpixel((10, 10))
