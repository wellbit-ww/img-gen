from __future__ import annotations

from PIL import Image

from app.config import Settings

_birefnet_model = None


class BiRefNetError(RuntimeError):
    pass


def remove_background_birefnet(image: Image.Image, settings: Settings) -> Image.Image:
    global _birefnet_model
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        model_id = "ZhengPeng7/BiRefNet"
        local = settings.models_dir / "BiRefNet"
        source = str(local) if local.exists() else model_id
        if _birefnet_model is None:
            _birefnet_model = (
                AutoModelForImageSegmentation.from_pretrained(
                    source,
                    token=settings.hf_token,
                    trust_remote_code=True,
                )
                .eval()
                .cuda()
                .half()
            )
        model = _birefnet_model
        image_size = (1024, 1024)
        transform_image = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = transform_image(image.convert("RGB")).unsqueeze(0).cuda().half()
        with torch.no_grad():
            preds = model(input_tensor)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(image.size, Image.Resampling.BILINEAR)
        rgba = image.convert("RGBA")
        rgba.putalpha(mask)
        return rgba
    except Exception as exc:
        if settings.engine_mode == "real":
            raise BiRefNetError(f"BiRefNet background removal failed: {exc}") from exc
        rgba = image.convert("RGBA")
        rgba.putalpha(128)
        return rgba
