from __future__ import annotations

import uvicorn

from app.worker.main import run_supervisor


def run_api() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


def run_worker() -> None:
    run_supervisor()
