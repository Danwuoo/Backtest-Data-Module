from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./okx_trading_platform.db"


def _database_url() -> str:
    return os.getenv("CONTROL_API_DATABASE_URL", DEFAULT_DATABASE_URL)


def _engine_kwargs(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


engine = create_engine(_database_url(), **_engine_kwargs(_database_url()))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def aget_db() -> Generator:
    for db in get_db():
        yield db
