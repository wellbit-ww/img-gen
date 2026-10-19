from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

EngineMode = Literal["mock", "real"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    engine_mode: EngineMode = "mock"
    redis_url: str = "redis://localhost:6379"
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"
    app_api_key: str | None = None
    hf_token: str | None = None
    workers_config: Path = Path("../config/workers.yaml")
    presets_dir: Path = Path("../config/presets")
    fonts_dir: Path = Path("../assets/fonts")
    cors_origins: str = "http://localhost:5173,http://localhost:8000"
    brand_colors: str = "#0B3D91,#FFFFFF,#1E88E5"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def brand_color_list(self) -> list[str]:
        return [c.strip() for c in self.brand_colors.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_path(base: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base / path).resolve()
