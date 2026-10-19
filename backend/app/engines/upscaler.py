from __future__ import annotations

from pathlib import Path

from PIL import Image


class UpscaleNotAvailableError(RuntimeError):
    pass


_upscale_model = None


def upscale_image(
    image: Image.Image,
    target_w: int,
    target_h: int,
    models_dir: Path,
    engine_mode: str = "mock",
) -> Image.Image:
    global _upscale_model
    model_path = models_dir / "upscale" / "4x-UltraSharp.pth"
    if not model_path.exists():
        if engine_mode == "real":
            raise UpscaleNotAvailableError(
                f"Upscale model missing at {model_path}. Download 4x-UltraSharp.pth first."
            )
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    try:
        import numpy as np
        import torch
        from spandrel import ModelLoader

        if _upscale_model is None:
            _upscale_model = ModelLoader().load_from_file(str(model_path)).eval().cuda()
        model = _upscale_model
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda().half()
        tile = 512
        overlap = 32
        h, w = tensor.shape[2], tensor.shape[3]
        out = torch.zeros(
            (1, 3, h * 4, w * 4),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        with torch.no_grad():
            for y in range(0, h, tile - overlap):
                for x in range(0, w, tile - overlap):
                    patch = tensor[:, :, y : y + tile, x : x + tile]
                    up = model(patch).clamp(0, 1)
                    oy, ox = y * 4, x * 4
                    out[:, :, oy : oy + up.shape[2], ox : ox + up.shape[3]] = up
        arr = (out.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype("uint8")
        out_img = Image.fromarray(arr)
        return out_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception as exc:
        if engine_mode == "real":
            raise UpscaleNotAvailableError(str(exc)) from exc
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)
