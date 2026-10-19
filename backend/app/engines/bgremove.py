from __future__ import annotations

from PIL import Image

from app.config import Settings


def remove_background_birefnet(image: Image.Image, settings: Settings) -> Image.Image:
    try:
        import torch
        from transformers import AutoModelForImageSegmentation, AutoProcessor

        model_id = "ZhengPeng7/BiRefNet"
        local = settings.models_dir / "BiRefNet"
        source = str(local) if local.exists() else model_id
        processor = AutoProcessor.from_pretrained(source, token=settings.hf_token)
        model = AutoModelForImageSegmentation.from_pretrained(
            source, token=settings.hf_token
        ).eval().cuda()
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            preds = model(**inputs)
        mask = preds[-1].sigmoid()[0, 0].cpu().numpy()
        rgba = image.convert("RGBA")
        alpha = Image.fromarray((mask * 255).astype("uint8"), mode="L")
        rgba.putalpha(alpha)
        return rgba
    except Exception:
        rgba = image.convert("RGBA")
        rgba.putalpha(128)
        return rgba
