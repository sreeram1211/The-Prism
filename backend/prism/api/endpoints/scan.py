"""
Prism Scan API endpoints — Phase 2 stub.

These endpoints will be fully implemented in Phase 2 (CLI + Web UI).
Currently returns HTTP 501 with a clear phase-readiness message.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/scan", tags=["Prism Scan"])

_STUB_RESPONSE = {
    "status": "not_implemented",
    "phase": 2,
    "message": (
        "Prism Scan is implemented in Phase 2. "
        "The 9-dimensional behavioral scan (sycophancy, hedging, calibration, "
        "depth, coherence, focus, specificity, verbosity, repetition) and the "
        "16-dimensional geometric separation ratio will be available then."
    ),
}


@router.post("/run", summary="[Phase 2] Run a behavioral scan")
async def run_scan() -> JSONResponse:
    return JSONResponse(status_code=501, content=_STUB_RESPONSE)


@router.get("/results/{scan_id}", summary="[Phase 2] Get scan results")
async def get_scan_results(scan_id: str) -> JSONResponse:
    return JSONResponse(status_code=501, content={**_STUB_RESPONSE, "scan_id": scan_id})
