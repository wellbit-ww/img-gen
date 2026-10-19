from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import Settings
from app.core.aspect import upscale_target_size
from app.engines.factory import create_engine
from app.engines.qwen_image import QwenImageEngine, _condition_images
from app.engines.utils_engine import UtilsEngine


def test_upscale_target_2k_preserves_aspect():
    w, h = upscale_target_size("2k", 2752, 1536)
    assert w == 1920
    assert h == 1056
    assert w <= 1920 and h <= 1080


def test_factory_unknown_model_real_raises():
    settings = Settings(engine_mode="real", hf_token=None)
    with pytest.raises(ValueError, match="Unknown model key"):
        create_engine(settings, "not-a-model")


def test_utils_remove_background_qwen_raises():
    settings = Settings(engine_mode="real")
    engine = UtilsEngine(settings)
    with pytest.raises(RuntimeError, match="Qwen background removal"):
        engine.remove_background(Image.new("RGB", (8, 8)), method="qwen")


def test_qwen_edit_forwards_strength_and_condition_images(monkeypatch):
    torch_stub = ModuleType("torch")

    class _Gen:
        def manual_seed(self, _seed: int) -> _Gen:
            return self

    torch_stub.Generator = lambda device: _Gen()  # noqa: ARG005
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    settings = Settings(engine_mode="mock")
    engine = QwenImageEngine(settings)
    engine._pipe = MagicMock()
    fake_result = MagicMock()
    fake_result.images = [Image.new("RGB", (64, 64))]
    engine._pipe.return_value = fake_result

    base = Image.new("RGB", (128, 128), color=(10, 20, 30))
    mask = Image.new("L", (128, 128), color=255)
    engine.edit(
        prompt="change sky",
        image=base,
        mask=mask,
        references=[],
        strength=0.65,
        steps=12,
        seed=1,
        on_step=lambda _s, _t: None,
        cancel_check=lambda: False,
    )
    kwargs = engine._pipe.call_args.kwargs
    assert kwargs["prompt"] == "change sky"
    assert kwargs["num_inference_steps"] == 12
    assert isinstance(kwargs["image"], Image.Image)


def test_qwen_remove_background_calls_edit():
    settings = Settings(engine_mode="mock")
    engine = QwenImageEngine(settings)
    with patch.object(engine, "edit", return_value=[Image.new("RGBA", (4, 4))]) as edit_mock:
        engine.remove_background(Image.new("RGB", (4, 4)), method="auto")
    edit_mock.assert_called_once()
    assert edit_mock.call_args.kwargs["strength"] == 1.0


def test_condition_images_includes_mask_overlay():
    base = Image.new("RGB", (32, 32), color=(0, 0, 0))
    mask = Image.new("L", (32, 32), color=200)
    refs = [Image.new("RGB", (16, 16))]
    images = _condition_images(base, mask, refs)
    assert len(images) == 2
    assert images[0].size == base.size


def test_true_cfg_scale_defaults_to_one():
    assert QwenImageEngine._true_cfg_scale(4.0, "") == 1.0
    assert QwenImageEngine._true_cfg_scale(4.0, "bad quality") == 4.0
