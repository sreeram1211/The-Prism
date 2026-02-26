"""
Prism Agent API endpoints — Phase 5.

POST   /api/v1/agent/chat                    — chat with Prism Agent
GET    /api/v1/agent/sessions/{id}/memory    — list memory entries
GET    /api/v1/agent/sessions/{id}/analytics — get α' analytics
DELETE /api/v1/agent/sessions/{id}           — clear a session
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from prism.agent.memory import MemoryEntry, PrismMemory
from prism.schemas.models import AgentChatRequest, AgentChatResponse

router = APIRouter(prefix="/agent", tags=["Prism Agent"])

_memory = PrismMemory()
# session_id → turn_index counter
_turn_counters: dict[str, int] = {}


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Chat with Prism Agent",
    description=(
        "Sends a message to the Prism Agent. The agent maintains per-session "
        "memory, retrieves relevant past context, and tracks RSI α' across turns."
    ),
    status_code=status.HTTP_200_OK,
)
def chat(body: AgentChatRequest) -> AgentChatResponse:
    session_id = body.session_id or str(uuid.uuid4())[:12]

    turn_index = _turn_counters.get(session_id, 0)
    _turn_counters[session_id] = turn_index + 1

    # --- Retrieve relevant memory context -----------------------------------
    memory_hits = 0
    context_snippets: list[str] = []

    if body.use_memory and turn_index > 0:
        hits = _memory.search(body.message, session_id=session_id, top_k=body.memory_top_k)
        memory_hits = len(hits)
        context_snippets = [h.entry.content for h in hits[:3]]

    # --- Generate reply -----------------------------------------------------
    reply = _generate_reply(body.message, session_id, turn_index, context_snippets)

    # --- Store user turn + assistant turn -----------------------------------
    _memory.store(MemoryEntry(
        entry_id=str(uuid.uuid4())[:8],
        session_id=session_id,
        turn_index=turn_index,
        role="user",
        content=body.message,
    ))
    _memory.store(MemoryEntry(
        entry_id=str(uuid.uuid4())[:8],
        session_id=session_id,
        turn_index=turn_index,
        role="assistant",
        content=reply,
    ))

    # --- Quality signal (heuristic: longer, specific replies = higher quality)
    quality = min(1.0, len(reply) / 500.0)
    _memory.record_quality_signal(session_id, quality)

    # --- α' -----------------------------------------------------------------
    alpha_prime = _memory.compute_alpha_prime(session_id)

    return AgentChatResponse(
        session_id=session_id,
        reply=reply,
        memory_hits=memory_hits,
        alpha_prime=alpha_prime,
    )


@router.get(
    "/sessions/{session_id}/memory",
    summary="List agent memory entries for a session",
)
def get_memory(session_id: str, limit: int = 20) -> dict[str, Any]:
    entries = _memory.all_entries(session_id)
    if not entries and session_id not in _turn_counters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "session_id": session_id},
        )
    return {
        "session_id": session_id,
        "entry_count": len(entries),
        "entries": [
            {
                "entry_id": e.entry_id,
                "turn_index": e.turn_index,
                "role": e.role,
                "content": e.content[:200] + ("…" if len(e.content) > 200 else ""),
                "timestamp": e.timestamp,
            }
            for e in entries[-limit:]
        ],
    }


@router.get(
    "/sessions/{session_id}/analytics",
    summary="Get α' analytics for a session",
)
def get_analytics(session_id: str) -> dict[str, Any]:
    if session_id not in _turn_counters:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "session_id": session_id},
        )
    alpha_prime = _memory.compute_alpha_prime(session_id)
    turn_count = _turn_counters.get(session_id, 0)
    return {
        "session_id": session_id,
        "turn_count": turn_count,
        "alpha_prime": alpha_prime,
        "alpha_prime_interpretation": _interpret_alpha(alpha_prime),
        "memory_entry_count": len(_memory.all_entries(session_id)),
    }


@router.delete(
    "/sessions/{session_id}",
    summary="Clear an agent session",
    status_code=status.HTTP_204_NO_CONTENT,
)
def clear_session(session_id: str) -> None:
    _memory._store.pop(session_id, None)
    _memory._quality_signals.pop(session_id, None)
    _turn_counters.pop(session_id, None)


# ---------------------------------------------------------------------------
# Mock response generator — Prism-domain knowledge base
# ---------------------------------------------------------------------------

_KNOWLEDGE: list[tuple[set[str], str]] = [
    (
        {"lora", "adapter", "rank", "peft", "finetune", "fine-tune"},
        "LoRA (Low-Rank Adaptation) injects trainable rank-decomposition matrices into "
        "each layer of a frozen pre-trained model. The rank *r* controls the expressivity "
        "of the adapter: higher rank captures more complex behavioral changes at the cost "
        "of more trainable parameters. Prism recommends ranks between 8–64 for behavioral "
        "fine-tuning, with alpha = 2×r as a starting point. Run `prism generate` to get "
        "architecture-specific recommendations derived from your scan results."
    ),
    (
        {"scan", "behavioral", "dimension", "sycophancy", "hedging", "calibration"},
        "Prism's 9-dimensional behavioral scan evaluates: **sycophancy** (agreement bias), "
        "**hedging** (qualification overload), **calibration** (confidence alignment), "
        "**depth** (reasoning elaboration), **coherence** (logical consistency), "
        "**focus** (topical discipline), **specificity** (concrete grounding), "
        "**verbosity** (length optimality), and **repetition** (lexical diversity). "
        "Each score is normalized 0–1. Use `prism scan <model_id>` to baseline your model."
    ),
    (
        {"manifold", "fiber", "space", "geometric", "separation", "coordinate"},
        "The Prism manifold projects model activations into a 16-dimensional fiber space "
        "where behavioral attractors cluster by family. The geometric separation ratio "
        "(GSR) measures how well-separated clusters are — a high GSR (>400×) indicates "
        "clean behavioral architecture that responds well to precision tuning. "
        "SSMs and attention models occupy distinct regions of this space."
    ),
    (
        {"monitor", "websocket", "telemetry", "activation", "layer", "pns"},
        "The Proprioceptive Nervous System (PNS) monitors 4 key layers: the input "
        "embedding layer (L0), a mid-early layer (L8), a mid-late layer (L16), and the "
        "penultimate layer (L24). These capture different abstraction levels. "
        "Reflex arc steering events occur when the model's own activation pattern "
        "overrides its learned reflex — visible as spikes in the monitor dashboard."
    ),
    (
        {"resolve", "architecture", "model", "attention", "ssm", "moe", "gqa"},
        "Prism resolves model architecture by fetching only `config.json` from HuggingFace. "
        "It identifies: attention family (standard transformer), SSM family (Mamba/RWKV), "
        "hybrid, or encoder-decoder. GQA (Grouped-Query Attention) models need fewer "
        "LoRA targets since key/value projections are shared. MoE (Mixture-of-Experts) "
        "models require special gating consideration during fine-tuning."
    ),
    (
        {"alpha", "prime", "rsi", "improvement", "acceleration", "learning"},
        "RSI α' (alpha-prime) is Prism's improvement-acceleration metric — the second "
        "derivative of your session quality signal over time. α' > 0 means your "
        "interactions are yielding faster improvements (accelerating), α' < 0 means "
        "diminishing returns (plateauing), α' ≈ 0 means steady linear improvement. "
        "Track it via `GET /api/v1/agent/sessions/{id}/analytics`."
    ),
    (
        {"install", "setup", "start", "run", "docker", "compose"},
        "To start Prism locally: `cd backend && uvicorn prism.main:app --reload --port 8000` "
        "for the API, and `cd frontend && npm run dev` for the UI at http://localhost:3000. "
        "Docker Compose is also available: `docker compose up` starts both services. "
        "The CLI is installed via `pip install -e backend/` — then `prism --help`."
    ),
    (
        {"training", "yaml", "config", "axolotl", "llamafactory", "dataset"},
        "The Prism-generated training YAML is compatible with LLaMA-Factory, Axolotl, "
        "and standard HuggingFace `transformers` PEFT workflows. It includes "
        "gradient checkpointing (for VRAM efficiency), cosine LR scheduling, "
        "and bf16 training flags. Replace `dataset: ./data/train.jsonl` with your "
        "JSONL file in Alpaca format. Prism sets the rank and alpha based on your "
        "behavioral target complexity automatically."
    ),
]

_FALLBACK_RESPONSE = (
    "I'm Prism Agent — your guide to the AI Behavioral Manifold tooling suite. "
    "I can help with: **resolve** (architecture profiling), **scan** "
    "(9-dimensional behavioral analysis), **generate** (LoRA adapter compilation), "
    "**monitor** (activation telemetry), and the underlying manifold geometry. "
    "What would you like to explore?"
)


def _generate_reply(
    message: str,
    session_id: str,
    turn_index: int,
    context_snippets: list[str],
) -> str:
    msg_tokens = set(
        word.lower().strip("?.!,;:")
        for word in message.split()
        if len(word) > 2
    )

    best_score = 0.0
    best_reply = ""

    for keywords, reply in _KNOWLEDGE:
        overlap = len(keywords & msg_tokens) / max(len(keywords), 1)
        if overlap > best_score:
            best_score = overlap
            best_reply = reply

    if best_score < 0.15:
        if turn_index == 0:
            return "Hello! " + _FALLBACK_RESPONSE
        return _FALLBACK_RESPONSE

    if context_snippets:
        context_note = (
            f"\n\n*Recalled from memory ({len(context_snippets)} hits):* "
            + " | ".join(s[:80] + "…" for s in context_snippets)
        )
        return best_reply + context_note

    return best_reply


def _interpret_alpha(alpha_prime: float | None) -> str:
    if alpha_prime is None:
        return "insufficient data (need ≥3 turns)"
    if alpha_prime > 0.02:
        return "accelerating improvement"
    if alpha_prime < -0.02:
        return "plateauing — consider rephrasing or new topic"
    return "steady linear improvement"
