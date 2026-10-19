from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class JobRecord(SQLModel, table=True):
    __tablename__ = "jobs"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_type: str
    status: str = "queued"
    queue: str | None = None
    params_json: str = "{}"
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)

    def set_params(self, data: dict[str, Any]) -> None:
        self.params_json = json.dumps(data, ensure_ascii=False)


class ImageRecord(SQLModel, table=True):
    __tablename__ = "images"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_id: str = Field(index=True)
    path: str
    prompt: str | None = None
    preset_id: str | None = None
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    favorite: bool = False
    has_overlay: bool = False
    params_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)

    def set_params(self, data: dict[str, Any]) -> None:
        self.params_json = json.dumps(data, ensure_ascii=False)


class AppSetting(SQLModel, table=True):
    __tablename__ = "settings"

    key: str = Field(primary_key=True)
    value: str


class LoraDataset(SQLModel, table=True):
    __tablename__ = "lora_datasets"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    name: str
    trigger_word: str
    path: str
    created_at: datetime = Field(default_factory=utc_now)


class LoraTrainJob(SQLModel, table=True):
    __tablename__ = "lora_trains"

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    dataset_id: str
    status: str = "queued"
    config_path: str
    output_path: str | None = None
    log: str = ""
    created_at: datetime = Field(default_factory=utc_now)
