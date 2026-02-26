"""
Prism Agent Memory — Phase 5.

In-memory implementation with simple TF-IDF-style keyword retrieval
and RSI α' (improvement acceleration) tracking.

The real implementation will use Qdrant / ChromaDB with dense embeddings.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single entry stored in the memory."""

    entry_id: str
    session_id: str
    turn_index: int
    role: str           # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySearchResult:
    entry: MemoryEntry
    score: float        # keyword overlap score, 0.0–1.0


class PrismMemory:
    """
    Phase 5 in-memory agent memory store with keyword-based retrieval
    and RSI α' improvement-acceleration tracking.
    """

    def __init__(self) -> None:
        # session_id → list of entries (in turn order)
        self._store: dict[str, list[MemoryEntry]] = defaultdict(list)
        # session_id → list of per-turn quality signals (0.0–1.0)
        self._quality_signals: dict[str, list[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        self._store[entry.session_id].append(entry)

    def all_entries(self, session_id: str) -> list[MemoryEntry]:
        return list(self._store.get(session_id, []))

    # ------------------------------------------------------------------
    # Retrieval — keyword overlap scoring
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[MemorySearchResult]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        candidates: list[MemoryEntry] = []
        if session_id:
            candidates = list(self._store.get(session_id, []))
        else:
            for entries in self._store.values():
                candidates.extend(entries)

        results: list[MemorySearchResult] = []
        for entry in candidates:
            score = _overlap_score(query_tokens, _tokenize(entry.content))
            if score > 0.0:
                results.append(MemorySearchResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # RSI α' — acceleration of improvement across turns
    # ------------------------------------------------------------------

    def record_quality_signal(self, session_id: str, signal: float) -> None:
        """Record a per-turn quality signal (0.0–1.0) for α' computation."""
        self._quality_signals[session_id].append(max(0.0, min(1.0, signal)))

    def compute_alpha_prime(self, session_id: str) -> float | None:
        """
        Compute α' — the second derivative of the quality signal.

        Returns None if fewer than 3 data points are available.
        α' > 0: improvement is accelerating
        α' < 0: improvement is decelerating (plateauing)
        α' ≈ 0: linear improvement
        """
        signals = self._quality_signals.get(session_id, [])
        if len(signals) < 3:
            return None

        # First differences (first derivative approximation)
        d1 = [signals[i + 1] - signals[i] for i in range(len(signals) - 1)]
        # Second differences (second derivative approximation)
        d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]

        if not d2:
            return None

        return round(sum(d2) / len(d2), 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "i", "you", "we", "this",
    "that", "was", "are", "be", "as", "its", "can", "do", "not", "me",
    "my", "your", "how", "what", "when", "where", "who", "which", "will",
    "have", "has", "had", "been", "he", "she", "they", "them", "their",
}


def _tokenize(text: str) -> set[str]:
    tokens = set()
    for word in text.lower().split():
        word = "".join(c for c in word if c.isalnum())
        if word and word not in _STOPWORDS and len(word) > 2:
            tokens.add(word)
    return tokens


def _overlap_score(q: set[str], d: set[str]) -> float:
    if not q or not d:
        return 0.0
    intersection = len(q & d)
    return intersection / (len(q) + 0.5 * len(d - q))
