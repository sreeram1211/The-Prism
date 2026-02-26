"""
Prism DB migrations — Phase 6.

We use SQLAlchemy's create_all() for the initial schema.
Called once at application startup from prism.main lifespan.
"""

from __future__ import annotations

import logging

from prism.db.engine import Base, engine
# Import models so their metadata is registered before create_all()
import prism.db.models  # noqa: F401

logger = logging.getLogger(__name__)


def create_all() -> None:
    """Create all tables that do not yet exist. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)
    logger.info("Prism DB tables ensured (SQLite).")
