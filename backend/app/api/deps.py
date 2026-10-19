from __future__ import annotations

from fastapi import Header, HTTPException, Request

from app.config import Settings, get_settings


def get_app_settings() -> Settings:
    return get_settings()


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = get_settings()
    if not settings.app_api_key:
        return
    if request.url.path in ("/api/health", "/openapi.json", "/docs", "/redoc"):
        return
    if x_api_key != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
