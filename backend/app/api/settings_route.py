from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.api.deps import get_app_settings
from app.config import Settings
from app.core.schemas import SettingsResponse, SettingsUpdate
from app.db.models import AppSetting
from app.db.session import get_session

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
def get_settings_route(
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> SettingsResponse:
    row = session.get(AppSetting, "brand_colors")
    colors = settings.brand_color_list
    if row:
        colors = json.loads(row.value)
    return SettingsResponse(
        brand_colors=colors,
        default_aspect="16:9",
        engine_mode=settings.engine_mode,
    )


@router.put("", response_model=SettingsResponse)
def update_settings(
    body: SettingsUpdate,
    settings: Settings = Depends(get_app_settings),
    session: Session = Depends(get_session),
) -> SettingsResponse:
    colors = body.brand_colors or settings.brand_color_list
    row = session.get(AppSetting, "brand_colors")
    if not row:
        row = AppSetting(key="brand_colors", value=json.dumps(colors))
    else:
        row.value = json.dumps(colors)
    session.add(row)
    session.commit()
    return SettingsResponse(
        brand_colors=colors,
        default_aspect="16:9",
        engine_mode=settings.engine_mode,
    )
