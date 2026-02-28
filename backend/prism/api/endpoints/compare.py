"""
Prism Comparison API endpoint — Phase 6.

POST /api/v1/compare  — diff two scan IDs, return dimensional deltas + composite distance
"""

from __future__ import annotations

import json
import math

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from prism.db import get_session
from prism.db.models import ScanRecord
from prism.schemas.models import CompareRequest, CompareResult, DimensionDelta, DimensionScore

router = APIRouter(prefix="/compare", tags=["Prism Compare"])


def _fetch_or_404(scan_id: str, db: Session) -> ScanRecord:
    rec = db.get(ScanRecord, scan_id)
    if rec is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "scan_id": scan_id},
        )
    return rec


def _scores_map(rec: ScanRecord) -> dict[str, float]:
    items = json.loads(rec.scores_json)
    return {item["dimension"]: item["score"] for item in items}


def _direction(delta: float) -> str:
    if abs(delta) < 0.01:
        return "neutral"
    return "improved" if delta > 0 else "regressed"


@router.post(
    "",
    response_model=CompareResult,
    summary="Compare two model scans side-by-side",
    description=(
        "Given two scan IDs, returns per-dimension deltas, L2 composite distance, "
        "and a winner declaration."
    ),
    status_code=status.HTTP_200_OK,
)
def compare_scans(
    body: CompareRequest,
    db: Session = Depends(get_session),
) -> CompareResult:
    rec_a = _fetch_or_404(body.scan_a, db)
    rec_b = _fetch_or_404(body.scan_b, db)

    map_a = _scores_map(rec_a)
    map_b = _scores_map(rec_b)

    # Union of all dimensions present in either scan
    all_dims = sorted(set(map_a) | set(map_b))

    deltas: list[DimensionDelta] = []
    sq_sum = 0.0
    for dim in all_dims:
        sa = map_a.get(dim, 0.0)
        sb = map_b.get(dim, 0.0)
        d = round(sb - sa, 6)
        sq_sum += d ** 2
        deltas.append(
            DimensionDelta(
                dimension=dim,
                score_a=sa,
                score_b=sb,
                delta=d,
                direction=_direction(d),
            )
        )

    composite_distance = round(math.sqrt(sq_sum), 6)

    # Winner: model with higher mean score across shared dimensions
    mean_a = sum(map_a.values()) / len(map_a) if map_a else 0.0
    mean_b = sum(map_b.values()) / len(map_b) if map_b else 0.0
    if abs(mean_a - mean_b) < 0.005:
        winner = "tie"
    elif mean_b > mean_a:
        winner = "b"
    else:
        winner = "a"

    return CompareResult(
        scan_id_a=rec_a.id,
        scan_id_b=rec_b.id,
        model_a=rec_a.model_id,
        model_b=rec_b.model_id,
        deltas=deltas,
        composite_distance=composite_distance,
        winner=winner,
    )
