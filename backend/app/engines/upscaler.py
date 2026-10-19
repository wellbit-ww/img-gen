from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


class UpscaleNotAvailableError(RuntimeError):
    pass


_upscale_model = None


def _load_model(model_path: Path):
    global _upscale_model
    from spandrel import ModelLoader

    if _upscale_model is None:
        _upscale_model = ModelLoader().load_from_file(str(model_path)).eval().cuda().half()
    return _upscale_model


def _upscale_patch_4x(model, tensor: Any) -> Any:
    import torch

    tile = 512
    overlap = 32
    stride = tile - overlap
    _, _, h, w = tensor.shape
    scale = 4
    out_h, out_w = h * scale, w * scale
    out = torch.zeros((1, 3, out_h, out_w), device=tensor.device, dtype=torch.float16)
    weights = torch.zeros((1, 1, out_h, out_w), device=tensor.device, dtype=torch.float16)
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile, h)
                x_end = min(x + tile, w)
                patch = tensor[:, :, y:y_end, x:x_end]
                up = model(patch).clamp(0, 1)
                oy, ox = y * scale, x * scale
                oh, ow = up.shape[2], up.shape[3]
                out[:, :, oy : oy + oh, ox : ox + ow] += up
                weights[:, :, oy : oy + oh, ox : ox + ow] += 1.0
    return (out / weights.clamp(min=1e-3)).clamp(0, 1)


def _tensor_from_image(image: Image.Image):
    from torchvision.transforms.functional import pil_to_tensor

    return pil_to_tensor(image.convert("RGB")).unsqueeze(0).cuda().half() / 255.0


def _image_from_tensor(tensor) -> Image.Image:
    arr = (tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def upscale_image(
    image: Image.Image,
    target_w: int,
    target_h: int,
    models_dir: Path,
    engine_mode: str = "mock",
) -> Image.Image:
    model_path = models_dir / "upscale" / "4x-UltraSharp.pth"
    if not model_path.exists():
        if engine_mode == "real":
            raise UpscaleNotAvailableError(
                f"Upscale model missing at {model_path}. Download 4x-UltraSharp.pth first."
            )
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    try:
        model = _load_model(model_path)
        current = image
        while max(current.width, current.height) * 4 < max(target_w, target_h):
            tensor = _tensor_from_image(current)
            tensor = _upscale_patch_4x(model, tensor)
            current = _image_from_tensor(tensor)
        return current.resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception as exc:
        if engine_mode == "real":
            raise UpscaleNotAvailableError(str(exc)) from exc
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)
