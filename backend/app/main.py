from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.deps import verify_api_key
from app.api.gallery import router as gallery_router
from app.api.generate import router as generate_router
from app.api.health import router as health_router
from app.api.jobs import router as jobs_router
from app.api.lora import router as lora_router
from app.api.presets import router as presets_router
from app.api.settings_route import router as settings_router
from app.config import get_settings
from app.db.session import init_db


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    init_db(settings.db_path)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Presentation Image Generator", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")
    app.include_router(
        generate_router,
        prefix="/api",
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        gallery_router,
        prefix="/api",
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        jobs_router,
        prefix="/api",
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        presets_router,
        prefix="/api",
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        settings_router,
        prefix="/api",
        dependencies=[Depends(verify_api_key)],
    )
    app.include_router(
        lora_router,
        prefix="/api",
        dependencies=[Depends(verify_api_key)],
    )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
