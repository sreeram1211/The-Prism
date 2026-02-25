"""
Prism Agent Memory — Phase 5 stub.

The full implementation provides:
  - Persistent vector memory backed by Qdrant or ChromaDB.
  - Semantic search over past conversation turns.
  - RSI engine α' (alpha-prime) metric: tracks the second derivative of
    user-defined "improvement" signal across sessions, indicating whether
    the agent's helpfulness is accelerating or plateauing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single entry stored in the vector DB."""

    entry_id: str
    session_id: str
    turn_index: int
    role: str                   # "user" | "assistant"
    content: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySearchResult:
    entry: MemoryEntry
    score: float                # cosine similarity


class PrismMemory:
    """
    Phase 5 placeholder.

    In Phase 5 this class will:
      1. Embed each conversation turn using a lightweight local embedding model.
      2. Store the embedding + text in the configured vector DB (Qdrant/Chroma).
      3. On each new turn, retrieve the top-k most semantically similar
         past entries to inject as context.
      4. Track α' by comparing session-level quality signals over time.
    """

    def __init__(self, backend: str = "qdrant") -> None:
        self.backend = backend

    def store(self, entry: MemoryEntry) -> None:
        raise NotImplementedError("Prism Agent memory is implemented in Phase 5.")

    def search(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        raise NotImplementedError("Prism Agent memory is implemented in Phase 5.")

    def compute_alpha_prime(self, session_id: str) -> float | None:
        """
        Compute α' — the acceleration of improvement across sessions.

        Returns None if there is insufficient history.
        """
        raise NotImplementedError("Prism Agent memory is implemented in Phase 5.")
