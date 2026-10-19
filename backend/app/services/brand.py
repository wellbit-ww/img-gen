from __future__ import annotations

import json

from sqlmodel import Session

from app.config import Settings
from app.db.models import AppSetting


def get_brand_colors(session: Session, settings: Settings) -> list[str]:
    row = session.get(AppSetting, "brand_colors")
    if row:
        try:
            colors = json.loads(row.value)
            if isinstance(colors, list) and colors:
                return [str(c) for c in colors]
        except json.JSONDecodeError:
            pass
    return settings.brand_color_list
