from __future__ import annotations

from typing import Any, Literal

MemoryStrategy = Literal["full", "text_encoder_offload", "fp8", "model_cpu_offload"]

MEMORY_FALLBACK_ORDER: list[MemoryStrategy] = [
    "full",
    "text_encoder_offload",
    "fp8",
    "model_cpu_offload",
]


def apply_memory_strategy(pipe: Any, strategy: MemoryStrategy) -> None:
    if strategy == "full":
        pipe.to("cuda")
        return
    if strategy == "model_cpu_offload" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return
    if strategy == "text_encoder_offload" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        return
    if strategy == "fp8":
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.float8 import float8_weight_only

            if hasattr(pipe, "transformer"):
                quantize_(pipe.transformer, float8_weight_only())
            pipe.to("cuda")
            return
        except Exception:
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
                return
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
