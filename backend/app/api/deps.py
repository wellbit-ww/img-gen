from __future__ import annotations

import hashlib

from fastapi import Header, HTTPException, Request

from app.config import Settings, get_settings

SESSION_COOKIE = "img_gen_session"


def get_app_settings() -> Settings:
    return get_settings()


def _session_ok(settings: Settings, cookie_value: str | None) -> bool:
    if not settings.app_api_key or not cookie_value:
        return False
    expected = hashlib.sha256(settings.app_api_key.encode("utf-8")).hexdigest()
    return cookie_value == expected


async def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings = get_settings()
    if not settings.app_api_key:
        return
    public_paths = (
        "/api/health",
        "/api/login",
        "/openapi.json",
        "/docs",
        "/redoc",
    )
    if request.url.path in public_paths:
        return
    if x_api_key == settings.app_api_key:
        return
    if _session_ok(settings, request.cookies.get(SESSION_COOKIE)):
        return
    raise HTTPException(status_code=401, detail="Invalid API key")
