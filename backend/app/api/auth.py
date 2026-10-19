from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import JSONResponse

from app.api.deps import SESSION_COOKIE
from app.config import Settings, get_settings

router = APIRouter(tags=["auth"])


def _session_value(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


@router.post("/login")
def login(
    password: str = Form(...),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    if not settings.app_api_key:
        return JSONResponse({"ok": True, "auth": "disabled"})
    if password != settings.app_api_key:
        raise HTTPException(status_code=401, detail="Invalid password")
    response = JSONResponse({"ok": True})
    response.set_cookie(
        key=SESSION_COOKIE,
        value=_session_value(settings.app_api_key),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    return response
