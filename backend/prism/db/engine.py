"""
SQLAlchemy engine + session factory for The Prism.

DB location defaults to ./prism.db; override with PRISM_DB_URL env var.
"""

from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_DB_URL: str = os.getenv("PRISM_DB_URL", "sqlite:///prism.db")

engine = create_engine(
    _DB_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

def get_session() -> Generator[Session, None, None]:
    """Yield a DB session; close it when the request finishes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
