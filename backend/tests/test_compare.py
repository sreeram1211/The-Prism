"""
Tests for Phase 6 Comparison API and CLI commands.

Coverage:
  - POST /api/v1/compare with two real scan IDs → 200 + CompareResult schema
  - deltas has 9 entries (one per dimension)
  - composite_distance >= 0
  - winner is one of "a" | "b" | "tie"
  - Comparing identical scans → composite_distance ≈ 0, all deltas ≈ 0, winner == "tie"
  - POST /api/v1/compare with nonexistent scan_id → 404
  - POST /api/v1/compare missing body fields → 422
  - direction values are valid
  - CLI: prism history exits 0
  - CLI: prism diff exits 0 after two scans
  - prism history --json outputs valid JSON
  - prism diff --json outputs valid JSON
  - prism history --limit N limits output
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from prism.cli.main import app as cli_app
from prism.db.engine import Base, engine
from prism.db.migrations import create_all


MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "meta-llama/Meta-Llama-3-8B"
cli_runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    create_all()
    yield
    Base.metadata.drop_all(bind=engine)
    create_all()  # leave tables intact for subsequent test modules


def _scan_id(client: TestClient, model: str = MODEL_A) -> str:
    resp = client.post("/api/v1/scan/run", json={"model_id": model})
    assert resp.status_code == 200, resp.text
    return resp.json()["scan_id"]


# ---------------------------------------------------------------------------
# Comparison API
# ---------------------------------------------------------------------------

class TestCompareAPI:
    def test_compare_two_scans_200(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        resp = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b})
        assert resp.status_code == 200

    def test_compare_result_schema(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        assert "scan_id_a" in data
        assert "scan_id_b" in data
        assert "model_a" in data
        assert "model_b" in data
        assert "deltas" in data
        assert "composite_distance" in data
        assert "winner" in data

    def test_compare_has_9_deltas(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        assert len(data["deltas"]) == 9

    def test_composite_distance_nonnegative(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        assert data["composite_distance"] >= 0.0

    def test_winner_valid_values(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        assert data["winner"] in ("a", "b", "tie")

    def test_direction_valid_values(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        for delta in data["deltas"]:
            assert delta["direction"] in ("improved", "regressed", "neutral")

    def test_compare_identical_scans(self, client: TestClient):
        """Same scan compared with itself → distance 0, winner tie."""
        id_a = _scan_id(client, MODEL_A)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_a}).json()
        assert data["composite_distance"] == pytest.approx(0.0, abs=1e-6)
        for delta in data["deltas"]:
            assert delta["delta"] == pytest.approx(0.0, abs=1e-6)
        assert data["winner"] == "tie"

    def test_compare_nonexistent_scan_a(self, client: TestClient):
        id_b = _scan_id(client, MODEL_B)
        resp = client.post("/api/v1/compare", json={"scan_a": "nonexistent", "scan_b": id_b})
        assert resp.status_code == 404

    def test_compare_nonexistent_scan_b(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        resp = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": "nonexistent"})
        assert resp.status_code == 404

    def test_compare_missing_body(self, client: TestClient):
        resp = client.post("/api/v1/compare", json={})
        assert resp.status_code == 422

    def test_delta_arithmetic(self, client: TestClient):
        """Each delta.delta == score_b - score_a."""
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        for d in data["deltas"]:
            expected = round(d["score_b"] - d["score_a"], 6)
            assert d["delta"] == pytest.approx(expected, abs=1e-4)

    def test_model_names_correct(self, client: TestClient):
        id_a = _scan_id(client, MODEL_A)
        id_b = _scan_id(client, MODEL_B)
        data = client.post("/api/v1/compare", json={"scan_a": id_a, "scan_b": id_b}).json()
        assert data["model_a"] == MODEL_A
        assert data["model_b"] == MODEL_B


# ---------------------------------------------------------------------------
# CLI: prism history
# ---------------------------------------------------------------------------

class TestCLIHistory:
    def test_history_exits_zero_empty(self):
        result = cli_runner.invoke(cli_app, ["history"])
        assert result.exit_code == 0

    def test_history_exits_zero_with_data(self, client: TestClient):
        _scan_id(client, MODEL_A)
        result = cli_runner.invoke(cli_app, ["history"])
        assert result.exit_code == 0

    def test_history_json_flag(self, client: TestClient):
        _scan_id(client, MODEL_A)
        result = cli_runner.invoke(cli_app, ["history", "--json"])
        assert result.exit_code == 0
        raw = result.stdout or result.output
        data = json.loads(raw)
        assert isinstance(data, list)
        assert data[0]["model_id"] == MODEL_A

    def test_history_limit(self, client: TestClient):
        for _ in range(5):
            _scan_id(client, MODEL_A)
        result = cli_runner.invoke(cli_app, ["history", "--limit", "2", "--json"])
        assert result.exit_code == 0
        raw = result.stdout or result.output
        data = json.loads(raw)
        assert len(data) <= 2


# ---------------------------------------------------------------------------
# CLI: prism diff
# ---------------------------------------------------------------------------

class TestCLIDiff:
    def test_diff_exits_zero(self, client: TestClient):
        _scan_id(client, MODEL_A)
        _scan_id(client, MODEL_B)
        result = cli_runner.invoke(cli_app, ["diff", MODEL_A, MODEL_B])
        assert result.exit_code == 0

    def test_diff_json_flag(self, client: TestClient):
        _scan_id(client, MODEL_A)
        _scan_id(client, MODEL_B)
        result = cli_runner.invoke(cli_app, ["diff", MODEL_A, MODEL_B, "--json"])
        assert result.exit_code == 0
        raw = result.stdout or result.output
        data = json.loads(raw)
        assert "model_a" in data
        assert "model_b" in data
        assert "composite_distance" in data
        assert data["composite_distance"] >= 0.0
