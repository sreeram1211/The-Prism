"""
SQLAlchemy ORM models for Prism persistence (Phase 6).

Tables
------
ScanRecord         — one row per POST /scan/run call
GenerateRecord     — one row per POST /generate/lora call
AgentSessionRecord — one row per active agent session (upserted on each turn)
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from prism.db.engine import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ScanRecord
# ---------------------------------------------------------------------------

class ScanRecord(Base):
    __tablename__ = "scan_records"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False)
    geo_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON blob: list[DimensionScore] serialised as JSON string
    scores_json: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ScanRecord id={self.id!r} model={self.model_id!r}>"


# ---------------------------------------------------------------------------
# GenerateRecord
# ---------------------------------------------------------------------------

class GenerateRecord(Base):
    __tablename__ = "generate_records"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    lora_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    lora_alpha: Mapped[float] = mapped_column(Float, nullable=False)
    lora_dropout: Mapped[float] = mapped_column(Float, nullable=False)
    # JSON blobs
    targets_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<GenerateRecord id={self.id!r} model={self.model_id!r}>"


# ---------------------------------------------------------------------------
# AgentSessionRecord
# ---------------------------------------------------------------------------

class AgentSessionRecord(Base):
    __tablename__ = "agent_session_records"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    turn_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<AgentSessionRecord id={self.id!r} turns={self.turn_count}>"
