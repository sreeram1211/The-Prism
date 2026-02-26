"""
Tests for Phase 6 persistence layer — SQLite via SQLAlchemy.

Coverage:
  - ScanRecord round-trips through DB (create → query → verify)
  - GenerateRecord round-trips
  - GET /api/v1/scan/history returns correct items
  - GET /api/v1/scan/results/{id} returns 200 after scan
  - GET /api/v1/scan/results/nonexistent returns 404
  - Pagination (limit/offset)
  - GET /api/v1/scan/history?model_id=<filter> filters correctly
  - POST /api/v1/scan/run persists and returns scan_id
  - GET /api/v1/scan/results/{id}/export?fmt=json → 200, Content-Disposition
  - GET /api/v1/scan/results/{id}/export?fmt=yaml → 200, yaml content-type
  - GET /api/v1/scan/results/{id}/export?fmt=peft → 200, contains behavioral_targets
  - GET /api/v1/dashboard/stats → 200, returns correct field names
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from prism.db.engine import Base, SessionLocal, engine
from prism.db.models import GenerateRecord, ScanRecord
from prism.db.migrations import create_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fresh_db():
    """Drop and recreate all tables before each test for isolation.
    Recreate in teardown so other test modules don't hit missing tables."""
    Base.metadata.drop_all(bind=engine)
    create_all()
    yield
    Base.metadata.drop_all(bind=engine)
    create_all()  # leave tables intact for subsequent test modules


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


MODEL_A = "mistralai/Mistral-7B-v0.1"
MODEL_B = "meta-llama/Meta-Llama-3-8B"


def _make_scan_record(model_id: str = MODEL_A, geo: float = 342.0) -> ScanRecord:
    scores = [
        {"dimension": "sycophancy", "score": 0.3, "raw_value": 0.3, "interpretation": "low"},
        {"dimension": "depth", "score": 0.8, "raw_value": 0.8, "interpretation": "high"},
    ]
    return ScanRecord(
        id=str(uuid.uuid4()),
        model_id=model_id,
        created_at=datetime.now(timezone.utc),
        duration_ms=120.5,
        geo_ratio=geo,
        scores_json=json.dumps(scores),
    )


# ---------------------------------------------------------------------------
# DB layer unit tests
# ---------------------------------------------------------------------------

class TestScanRecordPersistence:
    def test_create_and_query(self, db):
        rec = _make_scan_record()
        db.add(rec)
        db.commit()

        fetched = db.get(ScanRecord, rec.id)
        assert fetched is not None
        assert fetched.model_id == MODEL_A
        assert fetched.geo_ratio == pytest.approx(342.0)

    def test_scores_json_round_trip(self, db):
        rec = _make_scan_record()
        db.add(rec)
        db.commit()

        fetched = db.get(ScanRecord, rec.id)
        scores = json.loads(fetched.scores_json)
        assert len(scores) == 2
        assert scores[0]["dimension"] == "sycophancy"
        assert scores[1]["score"] == pytest.approx(0.8)

    def test_multiple_records_ordered_by_created_at(self, db):
        for i in range(3):
            rec = _make_scan_record(model_id=f"model-{i}")
            db.add(rec)
        db.commit()

        records = db.query(ScanRecord).order_by(ScanRecord.created_at.desc()).all()
        assert len(records) == 3


class TestGenerateRecordPersistence:
    def test_create_and_query(self, db):
        rec = GenerateRecord(
            id=str(uuid.uuid4()),
            model_id=MODEL_A,
            created_at=datetime.now(timezone.utc),
            lora_rank=16,
            lora_alpha=32.0,
            lora_dropout=0.05,
            targets_json=json.dumps({"depth": 0.8}),
            result_json=json.dumps({"status": "done"}),
        )
        db.add(rec)
        db.commit()

        fetched = db.get(GenerateRecord, rec.id)
        assert fetched is not None
        assert fetched.lora_rank == 16
        assert fetched.model_id == MODEL_A


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

class TestScanHistoryAPI:
    def test_run_scan_returns_scan_id(self, client: TestClient):
        resp = client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        assert resp.status_code == 200
        data = resp.json()
        assert "scan_id" in data
        assert data["scan_id"] is not None
        assert len(data["scan_id"]) == 36  # UUID format

    def test_history_empty_initially(self, client: TestClient):
        resp = client.get("/api/v1/scan/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_history_after_scan(self, client: TestClient):
        client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        resp = client.get("/api/v1/scan/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["model_id"] == MODEL_A

    def test_history_model_filter(self, client: TestClient):
        client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        client.post("/api/v1/scan/run", json={"model_id": MODEL_B})
        resp = client.get(f"/api/v1/scan/history?model_id={MODEL_A}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["model_id"] == MODEL_A

    def test_history_pagination(self, client: TestClient):
        for _ in range(5):
            client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        resp = client.get("/api/v1/scan/history?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_history_pagination_offset(self, client: TestClient):
        for _ in range(5):
            client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        resp = client.get("/api/v1/scan/history?limit=2&offset=4")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1  # only 1 left after offset 4


class TestGetScanResult:
    def test_get_result_after_scan(self, client: TestClient):
        run_resp = client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        scan_id = run_resp.json()["scan_id"]

        resp = client.get(f"/api/v1/scan/results/{scan_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == MODEL_A
        assert len(data["scores"]) == 9

    def test_get_result_nonexistent_returns_404(self, client: TestClient):
        resp = client.get("/api/v1/scan/results/nonexistent-uuid")
        assert resp.status_code == 404


class TestExportEndpoint:
    def _scan_id(self, client: TestClient, model: str = MODEL_A) -> str:
        resp = client.post("/api/v1/scan/run", json={"model_id": model})
        return resp.json()["scan_id"]

    def test_export_json(self, client: TestClient):
        scan_id = self._scan_id(client)
        resp = client.get(f"/api/v1/scan/results/{scan_id}/export?fmt=json")
        assert resp.status_code == 200
        assert "attachment" in resp.headers.get("content-disposition", "")
        data = resp.json()
        assert data["model_id"] == MODEL_A
        assert "scores" in data

    def test_export_yaml(self, client: TestClient):
        scan_id = self._scan_id(client)
        resp = client.get(f"/api/v1/scan/results/{scan_id}/export?fmt=yaml")
        assert resp.status_code == 200
        assert "yaml" in resp.headers.get("content-type", "")

    def test_export_peft(self, client: TestClient):
        scan_id = self._scan_id(client)
        resp = client.get(f"/api/v1/scan/results/{scan_id}/export?fmt=peft")
        assert resp.status_code == 200
        assert "behavioral_targets" in resp.text

    def test_export_invalid_fmt(self, client: TestClient):
        scan_id = self._scan_id(client)
        resp = client.get(f"/api/v1/scan/results/{scan_id}/export?fmt=xml")
        assert resp.status_code == 422


class TestDashboardStats:
    def test_dashboard_stats_empty(self, client: TestClient):
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_scans" in data
        assert "total_jobs" in data
        assert "total_sessions" in data
        assert "unique_models" in data
        assert "recent_scans" in data

    def test_dashboard_stats_after_scan(self, client: TestClient):
        client.post("/api/v1/scan/run", json={"model_id": MODEL_A})
        resp = client.get("/api/v1/dashboard/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_scans"] == 1
        assert data["unique_models"] == 1
        assert len(data["recent_scans"]) == 1
