"""
Prism Agent API endpoints — Phase 5 stub.

Full implementation in Phase 5: persistent vector memory (Qdrant/ChromaDB),
RSI engine with α' tracking, and chat interface.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/agent", tags=["Prism Agent"])

_STUB_RESPONSE = {
    "status": "not_implemented",
    "phase": 5,
    "message": (
        "Prism Agent is implemented in Phase 5. "
        "The autonomous chat interface with persistent vector memory and "
        "RSI engine α' (improvement acceleration) tracking will be available then. "
        "Requires a Pro license."
    ),
}


@router.post("/chat", summary="[Phase 5] Chat with Prism Agent")
async def chat() -> JSONResponse:
    return JSONResponse(status_code=501, content=_STUB_RESPONSE)


@router.get("/sessions/{session_id}/memory", summary="[Phase 5] Get agent memory")
async def get_memory(session_id: str) -> JSONResponse:
    return JSONResponse(status_code=501, content={**_STUB_RESPONSE, "session_id": session_id})


@router.get("/sessions/{session_id}/analytics", summary="[Phase 5] Get α' analytics")
async def get_analytics(session_id: str) -> JSONResponse:
    return JSONResponse(status_code=501, content={**_STUB_RESPONSE, "session_id": session_id})
