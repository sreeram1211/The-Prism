"""
Prism Monitor API endpoints — Phase 4 stub.

Full implementation in Phase 4: WebSocket-based real-time telemetry
from the 4-layer Proprioceptive Nervous System (<0.2ms/token).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/monitor", tags=["Prism Monitor"])

_STUB_RESPONSE = {
    "status": "not_implemented",
    "phase": 4,
    "message": (
        "Prism Monitor is implemented in Phase 4. "
        "Real-time WebSocket telemetry (<0.2ms/token) from the 4-layer "
        "Proprioceptive Nervous System, including reflex arc steering "
        "visualisation, will be available then. Requires a Pro license."
    ),
}


@router.post("/sessions", summary="[Phase 4] Start a monitor session")
async def start_session() -> JSONResponse:
    return JSONResponse(status_code=501, content=_STUB_RESPONSE)


@router.get("/sessions/{session_id}", summary="[Phase 4] Get session info")
async def get_session(session_id: str) -> JSONResponse:
    return JSONResponse(status_code=501, content={**_STUB_RESPONSE, "session_id": session_id})

# WebSocket endpoint placeholder — wired in Phase 4
# @router.websocket("/ws/{session_id}")
# async def telemetry_websocket(websocket: WebSocket, session_id: str): ...
