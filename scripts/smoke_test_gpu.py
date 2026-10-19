#!/usr/bin/env python3
"""Run on GPU server after deploy. Usage: uv run python scripts/smoke_test_gpu.py"""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import get_settings
from app.engines.factory import create_engine


def main() -> None:
    settings = get_settings()
    settings.engine_mode = "real"
    out = Path("benchmarks")
    out.mkdir(exist_ok=True)
    results = []
    for model in ("qwen-image-2.1-turbo", "utils"):
        engine = create_engine(settings, model)
        engine.load()
        t0 = time.time()
        if model == "utils":
            from PIL import Image

            img = Image.new("RGB", (512, 512), (100, 120, 200))
            engine.upscale(img, 1024, 1024)
        else:
            engine.generate(
                prompt="A minimal blue presentation slide background",
                width=1376,
                height=768,
                steps=8,
                guidance=1.0,
                num_images=1,
                seed=42,
                negative_prompt="",
                transparent=False,
                on_step=lambda s, t: None,
                cancel_check=lambda: False,
            )
        elapsed = time.time() - t0
        results.append({"model": model, "seconds": round(elapsed, 2)})
    path = out / f"smoke_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
