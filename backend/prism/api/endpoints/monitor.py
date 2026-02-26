"""
Prism Monitor API endpoints — Phase 5.

POST /api/v1/monitor/sessions              — create a monitor session
GET  /api/v1/monitor/sessions/{id}         — get session info
WS   /api/v1/monitor/ws/{session_id}       — stream activation frames
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from prism.monitor.telemetry import MonitorSession as TelemetrySession, PrismTelemetryStream
from prism.schemas.models import MonitorSession, MonitorSessionCreate

router = APIRouter(prefix="/monitor", tags=["Prism Monitor"])

_stream = PrismTelemetryStream()

# In-memory session store (Phase 5 mock)
_sessions: dict[str, TelemetrySession] = {}


@router.post(
    "/sessions",
    response_model=MonitorSession,
    summary="Create a monitor session",
    status_code=status.HTTP_201_CREATED,
)
async def create_session(body: MonitorSessionCreate) -> MonitorSession:
    session_id = str(uuid.uuid4())[:12]
    layers = body.layers or [0, 8, 16, 24]

    session = TelemetrySession(
        session_id=session_id,
        model_id=body.model_id,
        prompt=body.prompt,
        monitored_layers=layers,
    )
    _sessions[session_id] = session

    return MonitorSession(
        session_id=session_id,
        model_id=body.model_id,
        websocket_url=f"/api/v1/monitor/ws/{session_id}",
        telemetry_layers=layers,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=MonitorSession,
    summary="Get monitor session info",
)
async def get_session(session_id: str) -> MonitorSession:
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "session_id": session_id},
        )
    s = _sessions[session_id]
    return MonitorSession(
        session_id=s.session_id,
        model_id=s.model_id,
        websocket_url=f"/api/v1/monitor/ws/{s.session_id}",
        telemetry_layers=s.monitored_layers,
    )


@router.websocket("/ws/{session_id}")
async def telemetry_websocket(websocket: WebSocket, session_id: str) -> None:
    """
    Stream activation frames as newline-delimited JSON.

    Each frame is a JSON object matching ActivationFrame.as_dict().
    The stream ends after all tokens are emitted or the client disconnects.
    """
    if session_id not in _sessions:
        await websocket.close(code=4004, reason="session_not_found")
        return

    await websocket.accept()
    session = _sessions[session_id]

    try:
        async for frame in _stream.stream(session, max_tokens=80, tokens_per_second=6.0):
            await websocket.send_text(json.dumps(frame.as_dict()))
        # Signal stream end
        await websocket.send_text(json.dumps({"type": "done", "frame_count": session.frame_count}))
        await websocket.close()
    except WebSocketDisconnect:
        pass
