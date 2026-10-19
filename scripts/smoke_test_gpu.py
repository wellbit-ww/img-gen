#!/usr/bin/env python3
"""GPU smoke: each enabled worker model × presentation aspect formats."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.engines.factory import create_engine  # noqa: E402
from app.services.workers_config import load_workers_config  # noqa: E402

FORMATS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
    "4:3": (2400, 1792),
}


def main() -> None:
    settings = get_settings()
    settings.engine_mode = "real"
    out = BACKEND / "benchmarks"
    out.mkdir(exist_ok=True)
    workers = load_workers_config(settings)
    models = sorted({str(w.get("model", "mock")) for w in workers})
    results: list[dict] = []
    for model in models:
        if model in ("mock", "utils"):
            continue
        os.environ["WORKER_MODEL"] = model
        engine = create_engine(settings, model)
        engine.load()
        for label, (width, height) in FORMATS.items():
            import torch

            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            engine.generate(
                prompt="Minimal corporate presentation background, soft gradient",
                width=width,
                height=height,
                steps=8,
                guidance=1.0,
                num_images=1,
                seed=42,
                negative_prompt="",
                transparent=False,
                on_step=lambda _s, _t: None,
                cancel_check=lambda: False,
            )
            elapsed = time.time() - t0
            peak_mb = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
            results.append(
                {
                    "model": model,
                    "format": label,
                    "width": width,
                    "height": height,
                    "seconds": round(elapsed, 2),
                    "peak_cuda_mb": peak_mb,
                }
            )
    path = out / f"bench_{int(time.time())}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
