"""
Prism Dashboard API endpoint — Phase 6.

GET /api/v1/dashboard/stats  — live counters + recent scan feed
"""

from __future__ import annotations

import json
from datetime import timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from prism.db import get_session
from prism.db.models import AgentSessionRecord, GenerateRecord, ScanRecord
from prism.schemas.models import DashboardStats, DimensionScore, ScanHistoryItem

router = APIRouter(prefix="/dashboard", tags=["Prism Dashboard"])


def _record_to_history_item(rec: ScanRecord) -> ScanHistoryItem:
    scores = [DimensionScore(**item) for item in json.loads(rec.scores_json)]
    top = max(scores, key=lambda s: s.score)
    created_iso = rec.created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return ScanHistoryItem(
        scan_id=rec.id,
        model_id=rec.model_id,
        created_at=created_iso,
        duration_ms=rec.duration_ms,
        geometric_separation_ratio=rec.geo_ratio,
        top_score=top,
    )


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Get live dashboard statistics",
    status_code=status.HTTP_200_OK,
)
def dashboard_stats(db: Session = Depends(get_session)) -> DashboardStats:
    total_scans: int = db.query(ScanRecord).count()
    total_jobs: int = db.query(GenerateRecord).count()
    total_sessions: int = db.query(AgentSessionRecord).count()

    unique_models: int = (
        db.query(ScanRecord.model_id).distinct().count()
    )

    recent_records = (
        db.query(ScanRecord)
        .order_by(ScanRecord.created_at.desc())
        .limit(5)
        .all()
    )

    return DashboardStats(
        total_scans=total_scans,
        total_jobs=total_jobs,
        total_sessions=total_sessions,
        unique_models=unique_models,
        recent_scans=[_record_to_history_item(r) for r in recent_records],
    )
