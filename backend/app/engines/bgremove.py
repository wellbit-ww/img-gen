from __future__ import annotations

from PIL import Image

from app.config import Settings

_birefnet_model = None
_birefnet_processor = None


class BiRefNetError(RuntimeError):
    pass


def remove_background_birefnet(image: Image.Image, settings: Settings) -> Image.Image:
    global _birefnet_model, _birefnet_processor
    try:
        import torch
        from transformers import AutoModelForImageSegmentation, AutoProcessor

        model_id = "ZhengPeng7/BiRefNet"
        local = settings.models_dir / "BiRefNet"
        source = str(local) if local.exists() else model_id
        if _birefnet_processor is None:
            _birefnet_processor = AutoProcessor.from_pretrained(
                source,
                token=settings.hf_token,
                trust_remote_code=True,
            )
        if _birefnet_model is None:
            _birefnet_model = (
                AutoModelForImageSegmentation.from_pretrained(
                    source,
                    token=settings.hf_token,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
            )
        processor = _birefnet_processor
        model = _birefnet_model
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            preds = model(**inputs)
        mask = preds[-1].sigmoid()[0, 0].cpu().numpy()
        rgba = image.convert("RGBA")
        alpha = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        rgba.putalpha(alpha)
        return rgba
    except Exception as exc:
        if settings.engine_mode == "real":
            raise BiRefNetError(f"BiRefNet background removal failed: {exc}") from exc
        rgba = image.convert("RGBA")
        rgba.putalpha(128)
        return rgba
