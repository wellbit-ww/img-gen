from __future__ import annotations

from typing import Any, Literal

MemoryStrategy = Literal["full", "text_encoder_offload", "fp8", "model_cpu_offload"]

MEMORY_FALLBACK_ORDER: list[MemoryStrategy] = [
    "full",
    "text_encoder_offload",
    "fp8",
    "model_cpu_offload",
]


class MemoryStrategyError(RuntimeError):
    pass


def apply_memory_strategy(pipe: Any, strategy: MemoryStrategy) -> None:
    if strategy == "full":
        pipe.to("cuda")
        return
    if strategy == "model_cpu_offload" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        return
    if strategy == "text_encoder_offload":
        if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "to"):
            pipe.text_encoder.to("cpu")
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            return
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            return
    if strategy == "fp8":
        try:
            from torchao.quantization import quantize_
            from torchao.quantization.float8 import float8_weight_only
        except ImportError as exc:
            raise MemoryStrategyError(f"torchao is required for fp8 strategy: {exc}") from exc
        if not hasattr(pipe, "transformer"):
            raise MemoryStrategyError("FP8 strategy requires pipe.transformer")
        try:
            quantize_(pipe.transformer, float8_weight_only())
            pipe.to("cuda")
        except Exception as exc:
            raise MemoryStrategyError(f"FP8 quantization failed: {exc}") from exc
        return
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
