from __future__ import annotations

from pathlib import Path

from PIL import Image


def upscale_image(
    image: Image.Image,
    target_w: int,
    target_h: int,
    models_dir: Path,
) -> Image.Image:
    try:
        import numpy as np
        import torch
        from spandrel import ModelLoader

        model_path = models_dir / "upscale" / "4x-UltraSharp.pth"
        if not model_path.exists():
            return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        model = ModelLoader().load_from_file(str(model_path)).eval().cuda()
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            out = model(tensor).clamp(0, 1)
        arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        out_img = Image.fromarray(arr)
        return out_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    except Exception:
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)
