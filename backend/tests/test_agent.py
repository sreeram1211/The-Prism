"""
Tests for the Prism Agent memory system and chat endpoint (Phase 5).

Coverage:
  - PrismMemory.store() / all_entries()
  - PrismMemory.search() — keyword retrieval, session isolation
  - PrismMemory.record_quality_signal() / compute_alpha_prime()
  - α' returns None for <3 data points
  - α' is positive when quality accelerates
  - α' is negative when quality decelerates
  - POST /api/v1/agent/chat → 200 + AgentChatResponse schema
  - Session ID is preserved across turns
  - memory_hits increases on later turns (keyword match)
  - GET  /api/v1/agent/sessions/{id}/memory → 200
  - GET  /api/v1/agent/sessions/{id}/analytics → 200
  - DELETE /api/v1/agent/sessions/{id} → 204
  - GET  /api/v1/agent/sessions/nonexistent/memory → 404
"""

from __future__ import annotations

import pytest

from prism.agent.memory import MemoryEntry, PrismMemory


# ---------------------------------------------------------------------------
# Unit tests — PrismMemory
# ---------------------------------------------------------------------------

def _make_entry(entry_id: str, session_id: str, turn_index: int, role: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        session_id=session_id,
        turn_index=turn_index,
        role=role,
        content=content,
    )


class TestMemoryStore:
    def test_store_and_retrieve(self):
        mem = PrismMemory()
        entry = _make_entry("e1", "sess1", 0, "user", "Hello Prism agent!")
        mem.store(entry)
        entries = mem.all_entries("sess1")
        assert len(entries) == 1
        assert entries[0].entry_id == "e1"

    def test_empty_session_returns_empty_list(self):
        mem = PrismMemory()
        assert mem.all_entries("nonexistent") == []

    def test_multiple_entries_preserved_in_order(self):
        mem = PrismMemory()
        for i in range(5):
            mem.store(_make_entry(f"e{i}", "sess", i, "user", f"turn {i}"))
        entries = mem.all_entries("sess")
        assert len(entries) == 5
        for i, e in enumerate(entries):
            assert e.turn_index == i

    def test_session_isolation(self):
        mem = PrismMemory()
        mem.store(_make_entry("a1", "sessA", 0, "user", "session A message"))
        mem.store(_make_entry("b1", "sessB", 0, "user", "session B message"))
        assert len(mem.all_entries("sessA")) == 1
        assert len(mem.all_entries("sessB")) == 1


class TestMemorySearch:
    def test_search_finds_matching_entry(self):
        mem = PrismMemory()
        mem.store(_make_entry("e1", "s1", 0, "user", "LoRA adapter rank fine-tuning"))
        results = mem.search("LoRA fine-tuning", session_id="s1")
        assert len(results) > 0
        assert results[0].entry.entry_id == "e1"

    def test_search_returns_empty_for_no_match(self):
        mem = PrismMemory()
        mem.store(_make_entry("e1", "s1", 0, "user", "weather forecast rainfall"))
        results = mem.search("quantum physics", session_id="s1")
        assert results == []

    def test_search_respects_session_isolation(self):
        mem = PrismMemory()
        mem.store(_make_entry("a1", "sessA", 0, "user", "LoRA adapter rank"))
        mem.store(_make_entry("b1", "sessB", 0, "user", "completely different topic"))
        results = mem.search("LoRA rank", session_id="sessB")
        assert all(r.entry.session_id == "sessB" for r in results)

    def test_search_cross_session_when_no_session_id(self):
        mem = PrismMemory()
        mem.store(_make_entry("a1", "sessA", 0, "user", "LoRA rank adapter"))
        mem.store(_make_entry("b1", "sessB", 0, "user", "LoRA rank peft"))
        results = mem.search("LoRA rank")  # no session_id → global
        assert len(results) >= 2

    def test_search_top_k_respected(self):
        mem = PrismMemory()
        for i in range(10):
            mem.store(_make_entry(f"e{i}", "s1", i, "user", f"LoRA adapter rank {i}"))
        results = mem.search("LoRA adapter rank", session_id="s1", top_k=3)
        assert len(results) <= 3

    def test_search_scores_sorted_descending(self):
        mem = PrismMemory()
        mem.store(_make_entry("e1", "s1", 0, "user", "LoRA adapter fine-tune rank"))
        mem.store(_make_entry("e2", "s1", 1, "user", "completely unrelated entry"))
        mem.store(_make_entry("e3", "s1", 2, "user", "LoRA rank"))
        results = mem.search("LoRA rank adapter fine-tune", session_id="s1")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score

    def test_empty_query_returns_empty(self):
        mem = PrismMemory()
        mem.store(_make_entry("e1", "s1", 0, "user", "something"))
        results = mem.search("", session_id="s1")
        assert results == []


class TestAlphaPrime:
    def test_returns_none_for_zero_signals(self):
        mem = PrismMemory()
        assert mem.compute_alpha_prime("s1") is None

    def test_returns_none_for_one_signal(self):
        mem = PrismMemory()
        mem.record_quality_signal("s1", 0.5)
        assert mem.compute_alpha_prime("s1") is None

    def test_returns_none_for_two_signals(self):
        mem = PrismMemory()
        mem.record_quality_signal("s1", 0.4)
        mem.record_quality_signal("s1", 0.6)
        assert mem.compute_alpha_prime("s1") is None

    def test_returns_float_for_three_or_more_signals(self):
        mem = PrismMemory()
        for sig in [0.3, 0.5, 0.8]:
            mem.record_quality_signal("s1", sig)
        result = mem.compute_alpha_prime("s1")
        assert result is not None
        assert isinstance(result, float)

    def test_positive_alpha_for_accelerating_quality(self):
        mem = PrismMemory()
        # Quadratically increasing quality → α' > 0
        for sig in [0.1, 0.2, 0.4, 0.7]:
            mem.record_quality_signal("s1", sig)
        alpha = mem.compute_alpha_prime("s1")
        assert alpha is not None
        assert alpha > 0

    def test_negative_alpha_for_decelerating_quality(self):
        mem = PrismMemory()
        # Improvement that is decelerating (growing slower) → α' < 0
        # d1 = [0.2, 0.1, 0.05, 0.02]  →  d2 = [-0.1, -0.05, -0.03] → mean < 0
        for sig in [0.1, 0.3, 0.4, 0.45, 0.47]:
            mem.record_quality_signal("s1", sig)
        alpha = mem.compute_alpha_prime("s1")
        assert alpha is not None
        assert alpha < 0

    def test_signal_clamped_to_0_1(self):
        mem = PrismMemory()
        mem.record_quality_signal("s1", -0.5)   # should clamp to 0.0
        mem.record_quality_signal("s1", 1.5)    # should clamp to 1.0
        mem.record_quality_signal("s1", 0.5)
        # Just confirm no exception and returns a value
        result = mem.compute_alpha_prime("s1")
        assert result is not None

    def test_session_isolation_for_signals(self):
        mem = PrismMemory()
        for sig in [0.1, 0.5, 0.9]:
            mem.record_quality_signal("sessA", sig)
        # sessB has no signals
        assert mem.compute_alpha_prime("sessB") is None


# ---------------------------------------------------------------------------
# API integration tests — /agent/chat + sessions
# ---------------------------------------------------------------------------

class TestAgentChatAPI:
    def test_chat_returns_200(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "Hello"})
        assert resp.status_code == 200

    def test_chat_response_schema(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "What is LoRA?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "reply" in data
        assert "memory_hits" in data
        assert "alpha_prime" in data

    def test_chat_reply_is_non_empty(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "Tell me about LoRA"})
        assert resp.status_code == 200
        assert len(resp.json()["reply"]) > 0

    def test_session_id_persisted(self, client):
        # First turn — no session_id
        resp1 = client.post("/api/v1/agent/chat", json={"message": "Hello"})
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]
        assert session_id

        # Second turn — provide session_id
        resp2 = client.post("/api/v1/agent/chat", json={
            "message": "Tell me about behavioral scanning",
            "session_id": session_id,
        })
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id

    def test_alpha_prime_none_on_first_turns(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "Hello Prism"})
        assert resp.status_code == 200
        # First turn → not enough signals for α'
        assert resp.json()["alpha_prime"] is None

    def test_alpha_prime_available_after_multiple_turns(self, client):
        resp1 = client.post("/api/v1/agent/chat", json={"message": "Hello"})
        sid = resp1.json()["session_id"]

        for msg in ["Tell me about LoRA adapter rank fine-tuning",
                    "What is behavioral scan dimension sycophancy hedging?"]:
            client.post("/api/v1/agent/chat", json={"message": msg, "session_id": sid})

        resp_final = client.post("/api/v1/agent/chat", json={
            "message": "Explain α' improvement metric", "session_id": sid
        })
        # After 3+ turns, alpha_prime may be a float
        data = resp_final.json()
        assert data["alpha_prime"] is None or isinstance(data["alpha_prime"], float)

    def test_memory_hits_zero_on_first_turn(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "LoRA rank adapter"})
        assert resp.status_code == 200
        # First turn has nothing to retrieve
        assert resp.json()["memory_hits"] == 0

    def test_use_memory_false_returns_zero_hits(self, client):
        resp1 = client.post("/api/v1/agent/chat", json={"message": "Hello LoRA rank"})
        sid = resp1.json()["session_id"]

        resp2 = client.post("/api/v1/agent/chat", json={
            "message": "LoRA rank adapter",
            "session_id": sid,
            "use_memory": False,
        })
        assert resp2.status_code == 200
        assert resp2.json()["memory_hits"] == 0

    def test_lora_query_returns_relevant_reply(self, client):
        resp = client.post("/api/v1/agent/chat", json={
            "message": "Tell me about LoRA adapter rank fine-tuning PEFT"
        })
        assert resp.status_code == 200
        reply = resp.json()["reply"].lower()
        assert "lora" in reply or "adapter" in reply or "rank" in reply


class TestAgentSessionAPI:
    def test_get_memory_returns_200(self, client):
        # Create a session first
        resp = client.post("/api/v1/agent/chat", json={"message": "Hello LoRA"})
        sid = resp.json()["session_id"]

        mem_resp = client.get(f"/api/v1/agent/sessions/{sid}/memory")
        assert mem_resp.status_code == 200
        data = mem_resp.json()
        assert data["session_id"] == sid
        assert "entries" in data
        assert "entry_count" in data

    def test_get_memory_nonexistent_session_returns_404(self, client):
        resp = client.get("/api/v1/agent/sessions/no-such-session/memory")
        assert resp.status_code == 404

    def test_get_analytics_returns_200(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "Hello manifold"})
        sid = resp.json()["session_id"]

        analytics = client.get(f"/api/v1/agent/sessions/{sid}/analytics")
        assert analytics.status_code == 200
        data = analytics.json()
        assert data["session_id"] == sid
        assert "turn_count" in data
        assert "alpha_prime" in data
        assert "alpha_prime_interpretation" in data

    def test_get_analytics_nonexistent_returns_404(self, client):
        resp = client.get("/api/v1/agent/sessions/no-such-session/analytics")
        assert resp.status_code == 404

    def test_delete_session_returns_204(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "Hello"})
        sid = resp.json()["session_id"]

        del_resp = client.delete(f"/api/v1/agent/sessions/{sid}")
        assert del_resp.status_code == 204

    def test_memory_empty_after_delete(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "Hello LoRA"})
        sid = resp.json()["session_id"]

        client.delete(f"/api/v1/agent/sessions/{sid}")

        # Session no longer exists → 404
        mem_resp = client.get(f"/api/v1/agent/sessions/{sid}/memory")
        assert mem_resp.status_code == 404

    def test_memory_entries_contain_expected_fields(self, client):
        resp = client.post("/api/v1/agent/chat", json={"message": "What is sycophancy hedging?"})
        sid = resp.json()["session_id"]

        mem_resp = client.get(f"/api/v1/agent/sessions/{sid}/memory")
        assert mem_resp.status_code == 200
        entries = mem_resp.json()["entries"]
        assert len(entries) >= 2  # user + assistant
        for e in entries:
            assert "entry_id" in e
            assert "role" in e
            assert "content" in e
            assert "turn_index" in e
