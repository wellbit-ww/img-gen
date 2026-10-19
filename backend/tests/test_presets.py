from pathlib import Path

from app.core.presets import PresetStore, build_generation_params
from app.core.schemas import AspectRatio, GenerateRequest, Quality


def test_presets_load():
    presets_dir = Path(__file__).resolve().parents[2] / "config" / "presets"
    store = PresetStore(presets_dir)
    items = store.list()
    assert len(items) >= 8
    slide = store.get("slide-background")
    assert slide is not None
    req = GenerateRequest(prompt="Облачная платформа", preset_id="slide-background")
    params = build_generation_params(req, slide, ["#0B3D91", "#FFFFFF"])
    assert params["width"] == 2752
    assert params["height"] == 1536
    assert "Облачная платформа" in params["prompt"]


def test_draft_resolution():
    presets_dir = Path(__file__).resolve().parents[2] / "config" / "presets"
    store = PresetStore(presets_dir)
    slide = store.get("slide-background")
    req = GenerateRequest(
        prompt="test",
        preset_id="slide-background",
        quality=Quality.DRAFT,
        aspect_ratio=AspectRatio.RATIO_16_9,
    )
    params = build_generation_params(req, slide, [])
    assert params["width"] == 1376
    assert params["height"] == 768
