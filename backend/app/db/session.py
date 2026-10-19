from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from alembic.config import Config
from sqlalchemy import text
from sqlmodel import Session, create_engine

from alembic import command

_engine = None


def _sqlite_url(db_path: Path) -> str:
    resolved = db_path.expanduser().resolve()
    return f"sqlite:///{resolved.as_posix()}"


def _alembic_config(db_path: Path) -> Config:
    ini_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", _sqlite_url(db_path))
    cfg.attributes["configure_logger"] = False
    return cfg


def init_db(db_path: Path) -> None:
    global _engine
    resolved = db_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(_alembic_config(resolved), "head")
    _engine = create_engine(
        _sqlite_url(resolved),
        connect_args={"check_same_thread": False},
    )
    with _engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def get_session() -> Generator[Session, None, None]:
    if _engine is None:
        raise RuntimeError("Database not initialized")
    with Session(_engine) as session:
        yield session
