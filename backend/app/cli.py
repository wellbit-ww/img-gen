from __future__ import annotations

import uvicorn

from app.worker.main import run_supervisor


def run_api() -> None:
    from app.logging_setup import configure_logging

    configure_logging()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_config=None)


def run_worker() -> None:
    from app.logging_setup import configure_logging

    configure_logging()
    run_supervisor()
