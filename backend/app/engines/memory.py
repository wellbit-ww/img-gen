from __future__ import annotations

from typing import Any, Literal

MemoryStrategy = Literal["full", "text_encoder_offload", "fp8", "model_cpu_offload"]


def apply_memory_strategy(pipe: Any, strategy: MemoryStrategy) -> None:
    if strategy == "full":
        return
    if strategy == "model_cpu_offload" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return
    if strategy == "text_encoder_offload" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return
    if strategy == "fp8":
        # Weight-only FP8 is model-specific; fallback to cpu offload when unavailable.
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
