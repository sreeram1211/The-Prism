"""
Prism Generate API endpoints — Phase 3 stub.

Full implementation in Phase 3: LoRA adapter compilation from behavioral slider values.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/generate", tags=["Prism Generate"])

_STUB_RESPONSE = {
    "status": "not_implemented",
    "phase": 3,
    "message": (
        "Prism Generate is implemented in Phase 3. "
        "The LoRA adapter compilation engine (9-slider behavioral targeting via "
        "manifold geometry) will be available then. Requires a Pro license."
    ),
}


@router.post("/lora", summary="[Phase 3] Compile a precision LoRA adapter")
async def generate_lora() -> JSONResponse:
    return JSONResponse(status_code=501, content=_STUB_RESPONSE)


@router.get("/jobs/{job_id}", summary="[Phase 3] Poll a LoRA generation job")
async def get_job_status(job_id: str) -> JSONResponse:
    return JSONResponse(status_code=501, content={**_STUB_RESPONSE, "job_id": job_id})
