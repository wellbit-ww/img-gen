#!/usr/bin/env python3
"""VRAM and latency benchmark placeholder."""

from __future__ import annotations

import json
import time
from pathlib import Path

from app.config import get_settings
from app.engines.mock import MockEngine


def main() -> None:
    settings = get_settings()
    engine = MockEngine()
    engine.load()
    t0 = time.time()
    engine.generate(
        prompt="benchmark",
        width=1376,
        height=768,
        steps=8,
        guidance=1.0,
        num_images=1,
        seed=1,
        negative_prompt="",
        transparent=False,
        on_step=lambda _s, _t: None,
        cancel_check=lambda: False,
    )
    result = {
        "engine_mode": settings.engine_mode,
        "seconds": round(time.time() - t0, 3),
        "note": "Replace with torch.cuda.max_memory_allocated on GPU server",
    }
    out = Path("benchmarks")
    out.mkdir(exist_ok=True)
    path = out / f"bench_{int(time.time())}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
