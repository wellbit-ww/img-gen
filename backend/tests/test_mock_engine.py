from app.engines.mock import MockEngine


def test_mock_generate_respects_size():
    engine = MockEngine()
    images = engine.generate(
        prompt="test slide",
        width=640,
        height=360,
        steps=3,
        guidance=4.0,
        num_images=2,
        seed=1,
        negative_prompt="",
        transparent=False,
        on_step=lambda _s, _t: None,
        cancel_check=lambda: False,
    )
    assert len(images) == 2
    assert images[0].size == (640, 360)


def test_mock_rgba():
    engine = MockEngine()
    images = engine.generate(
        prompt="icon",
        width=256,
        height=256,
        steps=2,
        guidance=1.0,
        num_images=1,
        seed=None,
        negative_prompt="",
        transparent=True,
        on_step=lambda _s, _t: None,
        cancel_check=lambda: False,
    )
    assert images[0].mode == "RGBA"
