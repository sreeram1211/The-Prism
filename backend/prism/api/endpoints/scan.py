"""
Prism Scan API endpoint — Phase 2.

POST /api/v1/scan/run  — run a mock behavioral scan (9 dimensions)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from prism.scan.engine import ScanDimension as EngineScanDimension, get_scan_engine
from prism.schemas.models import ScanRequest, ScanResult, DimensionScore, ScanDimension

router = APIRouter(prefix="/scan", tags=["Prism Scan"])


@router.post(
    "/run",
    response_model=ScanResult,
    summary="Run a 9-dimensional behavioral scan",
    description=(
        "Runs the Prism mock behavioral probe engine across up to 9 dimensions. "
        "Phase 2 uses a seeded mock engine — real ROC-AUC-verified probes arrive in Phase 3."
    ),
    status_code=status.HTTP_200_OK,
)
def run_scan(body: ScanRequest) -> ScanResult:
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

    return ScanResult(
        model_id=report.model_id,
        scores=scores,
        geometric_separation_ratio=report.geometric_separation_ratio,
        scan_duration_ms=report.scan_duration_ms,
    )


@router.get("/results/{scan_id}", summary="Get scan results by ID", include_in_schema=False)
async def get_scan_results(scan_id: str) -> dict:
    # Phase 4 will wire this to persisted scan results
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail={
            "error": "not_implemented",
            "message": "Scan result retrieval by ID arrives in Phase 4.",
            "scan_id": scan_id,
        },
    )
