from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

_engine = None


def init_db(db_path: Path) -> None:
    global _engine
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(_engine)


def get_session() -> Generator[Session, None, None]:
    if _engine is None:
        raise RuntimeError("Database not initialized")
    with Session(_engine) as session:
        yield session
