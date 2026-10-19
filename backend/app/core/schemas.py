from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class AspectRatio(StrEnum):
    RATIO_16_9 = "16:9"
    RATIO_4_3 = "4:3"
    RATIO_1_1 = "1:1"
    RATIO_3_2 = "3:2"
    RATIO_2_3 = "2:3"
    RATIO_9_16 = "9:16"
    RATIO_3_4 = "3:4"


class Quality(StrEnum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"


class TextMode(StrEnum):
    NONE = "none"
    NATIVE = "native"
    OVERLAY = "overlay"


class ModelChoice(StrEnum):
    AUTO = "auto"
    QWEN = "qwen"
    ALT = "alt"


class OverlaySpec(BaseModel):
    title: str = ""
    subtitle: str = ""
    position: Literal["left", "center", "right", "top", "bottom"] = "left"
    font: Literal["inter", "montserrat"] = "inter"
    color: str = "#FFFFFF"


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str = ""
    preset_id: str | None = None
    aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9
    quality: Quality = Quality.STANDARD
    num_images: int = Field(default=1, ge=1, le=8)
    seed: int | None = None
    transparent: bool = False
    text_mode: TextMode = TextMode.NONE
    overlay: OverlaySpec | None = None
    enhance_prompt: bool = False
    model: ModelChoice = ModelChoice.AUTO
    lora_id: str | None = None
    lora_scale: float = Field(default=0.8, ge=0.0, le=2.0)


class EditRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    image_id: str | None = None
    reference_image_ids: list[str] = Field(default_factory=list, max_length=10)
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    quality: Quality = Quality.STANDARD
    seed: int | None = None
    preset_id: str | None = None


class UpscaleRequest(BaseModel):
    image_id: str
    target: Literal["4k", "2k"] = "4k"


class RemoveBgRequest(BaseModel):
    image_id: str
    method: Literal["auto", "qwen", "birefnet"] = "auto"


class BatchLine(BaseModel):
    prompt: str
    overlay_title: str | None = None


class BatchRequest(BaseModel):
    lines: list[BatchLine] = Field(min_length=1, max_length=50)
    preset_id: str | None = None
    aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9
    quality: Quality = Quality.DRAFT
    num_images: int = Field(default=1, ge=1, le=4)
    text_mode: TextMode = TextMode.OVERLAY


class OverlayApplyRequest(BaseModel):
    overlay: OverlaySpec


class EnhancePromptRequest(BaseModel):
    prompt: str
    preset_id: str | None = None
    aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9
    text_mode: TextMode = TextMode.NONE


class EnhancePromptResponse(BaseModel):
    prompt: str
    negative_prompt: str = ""
    overlay_title: str | None = None
    notes: str | None = None
    warning: str | None = None


class JobCreateResponse(BaseModel):
    job_id: str
    queue: str | None = None
    notice: str | None = None


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    job_type: str
    error: str | None = None
    image_ids: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class ImageMeta(BaseModel):
    id: str
    job_id: str
    path: str
    url: str
    prompt: str | None = None
    preset_id: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    favorite: bool = False
    has_overlay: bool = False
    created_at: str
    params: dict[str, Any] = Field(default_factory=dict)


class GalleryResponse(BaseModel):
    items: list[ImageMeta]
    total: int
    page: int
    page_size: int


class PresetInfo(BaseModel):
    id: str
    name: str
    default_aspect: str
    transparent: bool
    text_mode: str
    description: str = ""


class ModelInfo(BaseModel):
    id: str
    name: str
    capabilities: list[str]
    queues: list[str]
    enabled: bool


class HealthResponse(BaseModel):
    status: str
    engine_mode: str
    redis: bool
    workers: list[dict[str, Any]]


class SettingsResponse(BaseModel):
    brand_colors: list[str]
    default_aspect: str
    engine_mode: str


class SettingsUpdate(BaseModel):
    brand_colors: list[str] | None = None


class LoraDatasetCreate(BaseModel):
    name: str
    trigger_word: str = "brandstyle"


class LoraTrainRequest(BaseModel):
    dataset_id: str
    rank: int = Field(default=16, ge=4, le=64)
    steps: int = Field(default=2000, ge=500, le=10000)
    learning_rate: float = 1e-4
