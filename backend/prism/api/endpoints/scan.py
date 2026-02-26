"""
Prism Scan API endpoint — Phase 2 + Phase 6.

POST /api/v1/scan/run              — run a behavioral scan and persist it
GET  /api/v1/scan/history          — list past scan records
GET  /api/v1/scan/results/{id}     — fetch a single scan record
GET  /api/v1/scan/results/{id}/export — download scan as JSON / YAML / PEFT
"""

from __future__ import annotations

import json
import uuid
from datetime import timezone

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from prism.db import get_session
from prism.db.models import ScanRecord
from prism.scan.engine import ScanDimension as EngineScanDimension, get_scan_engine
from prism.schemas.models import (
    DimensionScore,
    ScanDimension,
    ScanHistoryItem,
    ScanHistoryResponse,
    ScanRequest,
    ScanResult,
)

router = APIRouter(prefix="/scan", tags=["Prism Scan"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scores_to_json(scores: list[DimensionScore]) -> str:
    return json.dumps([s.model_dump(mode="json") for s in scores])


def _scores_from_json(raw: str) -> list[DimensionScore]:
    return [DimensionScore(**item) for item in json.loads(raw)]


def _record_to_history_item(rec: ScanRecord) -> ScanHistoryItem:
    scores = _scores_from_json(rec.scores_json)
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


def _fetch_record_or_404(scan_id: str, db: Session) -> ScanRecord:
    rec = db.get(ScanRecord, scan_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "scan_id": scan_id},
        )
    return rec


# ---------------------------------------------------------------------------
# POST /scan/run
# ---------------------------------------------------------------------------

@router.post(
    "/run",
    response_model=ScanResult,
    summary="Run a 9-dimensional behavioral scan",
    description=(
        "Runs the Prism mock behavioral probe engine across up to 9 dimensions. "
        "Results are persisted to SQLite and retrievable by scan_id."
    ),
    status_code=status.HTTP_200_OK,
)
def run_scan(
    body: ScanRequest,
    db: Session = Depends(get_session),
) -> ScanResult:
    # Map schema ScanDimension → engine ScanDimension
    selected: list[EngineScanDimension] | None = None
    if body.dimensions:
        try:
            selected = [EngineScanDimension(d.value) for d in body.dimensions]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "invalid_dimension", "message": str(exc)},
            ) from exc

    engine = get_scan_engine(mock=True)
    report = engine.scan(body.model_id, dimensions=selected)

    scores = [
        DimensionScore(
            dimension=ScanDimension(ds.dimension.value),
            score=ds.score,
            raw_value=ds.raw_value,
            interpretation=ds.interpretation,
        )
        for ds in report.scores
    ]

    scan_id = str(uuid.uuid4())

    # Persist to DB
    rec = ScanRecord(
        id=scan_id,
        model_id=report.model_id,
        duration_ms=report.scan_duration_ms,
        geo_ratio=report.geometric_separation_ratio,
        scores_json=_scores_to_json(scores),
    )
    db.add(rec)
    db.commit()

    return ScanResult(
        model_id=report.model_id,
        scan_id=scan_id,
        scores=scores,
        geometric_separation_ratio=report.geometric_separation_ratio,
        scan_duration_ms=report.scan_duration_ms,
    )


# ---------------------------------------------------------------------------
# GET /scan/history
# ---------------------------------------------------------------------------

@router.get(
    "/history",
    response_model=ScanHistoryResponse,
    summary="List past scan results",
)
def scan_history(
    model_id: str | None = Query(default=None, description="Filter by model ID"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
) -> ScanHistoryResponse:
    q = db.query(ScanRecord)
    if model_id:
        q = q.filter(ScanRecord.model_id == model_id)
    total: int = q.count()
    records = q.order_by(ScanRecord.created_at.desc()).offset(offset).limit(limit).all()
    return ScanHistoryResponse(
        items=[_record_to_history_item(r) for r in records],
        total=total,
        limit=limit,
        offset=offset,
    )


# ---------------------------------------------------------------------------
# GET /scan/results/{scan_id}
# ---------------------------------------------------------------------------

@router.get(
    "/results/{scan_id}",
    response_model=ScanResult,
    summary="Fetch a scan result by ID",
)
def get_scan_result(
    scan_id: str,
    db: Session = Depends(get_session),
) -> ScanResult:
    rec = _fetch_record_or_404(scan_id, db)
    scores = _scores_from_json(rec.scores_json)
    return ScanResult(
        model_id=rec.model_id,
        scan_id=rec.id,
        scores=scores,
        geometric_separation_ratio=rec.geo_ratio,
        scan_duration_ms=rec.duration_ms,
    )


# ---------------------------------------------------------------------------
# GET /scan/results/{scan_id}/export
# ---------------------------------------------------------------------------

@router.get(
    "/results/{scan_id}/export",
    summary="Export scan result as JSON, YAML, or PEFT behavioral profile",
)
def export_scan_result(
    scan_id: str,
    fmt: str = Query(default="json", description="Export format: json | yaml | peft"),
    db: Session = Depends(get_session),
) -> Response:
    rec = _fetch_record_or_404(scan_id, db)
    scores = _scores_from_json(rec.scores_json)
    created_iso = rec.created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if fmt == "json":
        payload = {
            "scan_id": rec.id,
            "model_id": rec.model_id,
            "created_at": created_iso,
            "scan_duration_ms": rec.duration_ms,
            "geometric_separation_ratio": rec.geo_ratio,
            "scores": [s.model_dump(mode="json") for s in scores],
        }
        return Response(
            content=json.dumps(payload, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="scan_{scan_id[:8]}.json"'},
        )

    if fmt == "yaml":
        payload = {
            "scan_id": rec.id,
            "model_id": rec.model_id,
            "created_at": created_iso,
            "scan_duration_ms": rec.duration_ms,
            "geometric_separation_ratio": rec.geo_ratio,
            "scores": [s.model_dump(mode="json") for s in scores],
        }
        return Response(
            content=yaml.dump(payload, sort_keys=False, allow_unicode=True),
            media_type="application/yaml",
            headers={"Content-Disposition": f'attachment; filename="scan_{scan_id[:8]}.yaml"'},
        )

    if fmt == "peft":
        behavioral_targets = {s.dimension.value: round(s.score, 4) for s in scores}
        peft_payload = {
            "# Prism behavioral fingerprint — use with prism generate": None,
            "model_id": rec.model_id,
            "scan_id": rec.id,
            "behavioral_targets": behavioral_targets,
        }
        # Build clean YAML manually so the comment appears at the top
        lines = [
            "# Prism behavioral fingerprint — use with prism generate",
            f"model_id: {rec.model_id}",
            f"scan_id: {rec.id}",
            "behavioral_targets:",
        ]
        for dim, val in behavioral_targets.items():
            lines.append(f"  {dim}: {val}")
        content = "\n".join(lines) + "\n"
        return Response(
            content=content,
            media_type="application/yaml",
            headers={
                "Content-Disposition": f'attachment; filename="behavioral_profile_{scan_id[:8]}.yaml"'
            },
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": "invalid_format", "message": "fmt must be one of: json, yaml, peft"},
    )
