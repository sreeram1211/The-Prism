"""
Prism persistence layer — SQLite via SQLAlchemy 2.0.

Public API:
    get_session()  — FastAPI dependency that yields a DB session
"""

from prism.db.engine import get_session

__all__ = ["get_session"]
