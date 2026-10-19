from __future__ import annotations

from app.core.schemas import AspectRatio, Quality

# Qwen-Image-2.1 recommended resolutions (full quality).
FULL_RESOLUTIONS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.RATIO_1_1: (2048, 2048),
    AspectRatio.RATIO_4_3: (2400, 1792),
    AspectRatio.RATIO_3_4: (1792, 2400),
    AspectRatio.RATIO_3_2: (2528, 1696),
    AspectRatio.RATIO_2_3: (1696, 2528),
    AspectRatio.RATIO_16_9: (2752, 1536),
    AspectRatio.RATIO_9_16: (1536, 2752),
}


def _round16(value: int) -> int:
    return max(16, (value // 16) * 16)


def resolve_size(aspect: AspectRatio, quality: Quality) -> tuple[int, int]:
    w, h = FULL_RESOLUTIONS[aspect]
    if quality != Quality.DRAFT:
        return w, h
    return _round16(w // 2), _round16(h // 2)


def upscale_target_size(target: str, src_w: int, src_h: int) -> tuple[int, int]:
    if target == "2k":
        max_w, max_h = 1920, 1080
    else:
        max_w, max_h = 3840, 2160
    scale = min(max_w / src_w, max_h / src_h)
    return _round16(int(src_w * scale)), _round16(int(src_h * scale))
