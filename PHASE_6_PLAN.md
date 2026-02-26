# Phase 6 Plan — Behavioral Fingerprinting + Comparison Engine

## Theme

Every scan, job, and session disappears on restart today.
Phase 6 makes The Prism *stateful*: results are persisted to SQLite, models
can be compared side-by-side on an overlay radar, and a live dashboard
replaces the static landing page.  The CLI gains `history` and `diff`.

---

## Scope

| # | Area | What changes |
|---|------|-------------|
| 6.1 | SQLite persistence layer | New `db/` package — SQLAlchemy models + migrations for scan results, generate jobs, agent sessions |
| 6.2 | Scan history API | `GET /api/v1/scan/history`, `GET /api/v1/scan/results/{scan_id}` (was 501 stub) |
| 6.3 | Comparison API | `POST /api/v1/compare` — diff two scan_ids, return dimensional deltas + composite distance |
| 6.4 | Frontend: History page | `/history` — searchable table of past scans with timestamps, click-to-expand |
| 6.5 | Frontend: Compare page | `/compare` — pick two model scans, render overlaid radar + delta heatmap |
| 6.6 | Frontend: Dashboard (home page) | Replace static landing with live counters (scans, jobs, sessions) + sparkline |
| 6.7 | Export endpoint | `GET /api/v1/scan/results/{scan_id}/export?fmt=json\|yaml\|peft` |
| 6.8 | CLI: `prism history` | List recent scans in a Rich table; `--limit N`, `--model filter` |
| 6.9 | CLI: `prism diff` | `prism diff <model_a> <model_b>` — Rich side-by-side delta table |
| 6.10 | Home page / nav update | Add History + Compare nav items; bump phase indicator to "Phase 6" |
| 6.11 | Tests | ≥ 25 new tests → total ≥ 170 passing |

---

## 6.1 — SQLite Persistence Layer

### New files

**`backend/prism/db/__init__.py`**
Empty; exposes `get_session`.

**`backend/prism/db/engine.py`**
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DB_URL = "sqlite:///prism.db"   # respects PRISM_DB_URL env var
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase): ...

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**`backend/prism/db/models.py`**
Three ORM tables:

```
ScanRecord
  id          TEXT PK (uuid4)
  model_id    TEXT NOT NULL
  created_at  DATETIME
  duration_ms FLOAT
  geo_ratio   FLOAT
  scores_json TEXT   -- JSON blob of list[DimensionScore]

GenerateRecord
  id           TEXT PK
  model_id     TEXT NOT NULL
  created_at   DATETIME
  lora_rank    INT
  lora_alpha   FLOAT
  lora_dropout FLOAT
  targets_json TEXT  -- behavioral targets
  result_json  TEXT  -- full GenerateLoRAResult

AgentSessionRecord
  id          TEXT PK
  created_at  DATETIME
  turn_count  INT
  last_active DATETIME
```

**`backend/prism/db/migrations.py`**
`create_all()` called at app startup in `main.py`.

---

## 6.2 — Scan History API

### Modified: `backend/prism/api/endpoints/scan.py`

- `POST /api/v1/scan/run` — after returning result, persist a `ScanRecord` via `db.add()`
- `GET /api/v1/scan/history` — query params: `model_id` (optional filter), `limit` (default 20), `offset`; returns `ScanHistoryResponse`
- `GET /api/v1/scan/results/{scan_id}` — was 501 stub; now fetches from DB

### New schemas (add to `schemas/models.py`):

```python
class ScanHistoryItem(BaseModel):
    scan_id: str
    model_id: str
    created_at: str   # ISO-8601
    duration_ms: float
    geometric_separation_ratio: float
    top_score: DimensionScore

class ScanHistoryResponse(BaseModel):
    items: list[ScanHistoryItem]
    total: int
    limit: int
    offset: int
```

---

## 6.3 — Comparison API

### New file: `backend/prism/api/endpoints/compare.py`

```
POST /api/v1/compare
  body: { scan_a: str, scan_b: str }   # scan_ids

Returns: CompareResult
  model_a, model_b,
  deltas: list[DimensionDelta]
    dimension, score_a, score_b, delta, direction ("improved"|"regressed"|"neutral")
  composite_distance: float   # L2 distance in 9-dim score space
  winner: "a" | "b" | "tie"   # which model is "better" by composite score
```

Register router in `api/router.py`.

### New schemas:

```python
class DimensionDelta(BaseModel):
    dimension: str
    score_a: float
    score_b: float
    delta: float        # score_b - score_a
    direction: str      # "improved" | "regressed" | "neutral"

class CompareResult(BaseModel):
    scan_id_a: str
    scan_id_b: str
    model_a: str
    model_b: str
    deltas: list[DimensionDelta]
    composite_distance: float
    winner: str         # "a" | "b" | "tie"
```

---

## 6.4 — Frontend: History Page

### New file: `frontend/app/history/page.tsx`

Layout:
- Search bar (filter by model_id)
- Paginated table:
  - Model ID | Scan date | Duration | Geo-ratio | Top dim score
  - Row click → expands inline radar sparkline (reuse recharts `RadarChart`)
- "Compare" checkbox on two rows → redirects to `/compare?a=<id>&b=<id>`
- "Export" icon per row → downloads JSON fingerprint

API calls:
- `GET /api/v1/scan/history?limit=20&offset=0`
- `GET /api/v1/scan/results/{id}` (for expansion)

---

## 6.5 — Frontend: Compare Page

### New file: `frontend/app/compare/page.tsx`

Layout (two-column):
- **Left panel**: two model pickers (dropdown populated from history)
  - "Load from scan ID" text input as fallback
  - Compare button
- **Right panel** (once loaded):
  - Overlaid `RadarChart` (Model A in violet, Model B in cyan, semi-transparent fills)
  - Delta heatmap (9 cells: green = improved, red = regressed, grey = neutral)
  - Composite distance badge + winner chip
  - "Export diff" button → downloads `diff_<a>_vs_<b>.json`

API calls:
- `POST /api/v1/compare` with `{ scan_a, scan_b }`

---

## 6.6 — Frontend: Dashboard (Home Page)

### Modified: `frontend/app/page.tsx`

Replace static hero + 4 feature cards with a live dashboard:

**Top row — counters strip** (fetched from `/api/v1/dashboard/stats`):
```
[ Total Scans: 14 ]  [ Generate Jobs: 6 ]  [ Agent Sessions: 3 ]  [ Models Seen: 8 ]
```

**Recent activity feed** (last 5 scans from history API):
```
Model                Date       Geo-ratio   Top Dim
─────────────────────────────────────────────────────
meta-llama/Llama...  2 min ago  342×        depth 0.82
mistralai/Mistral…   1 hr ago   289×        calibration 0.79
```

**Quick-action buttons**:
- Resolve a model → `/resolve`
- Run a scan → `/scan`
- Compare models → `/compare`

**New backend endpoint**: `GET /api/v1/dashboard/stats`
Returns: `{ total_scans, total_jobs, total_sessions, unique_models, recent_scans: list[ScanHistoryItem] }`

New file: `backend/prism/api/endpoints/dashboard.py`

---

## 6.7 — Export Endpoint

### Modified: `backend/prism/api/endpoints/scan.py`

`GET /api/v1/scan/results/{scan_id}/export`
Query param: `fmt` = `json` | `yaml` | `peft`

- `json` — full ScanResult as JSON (`Content-Disposition: attachment`)
- `yaml` — YAML serialization of same
- `peft` — generates a minimal `behavioral_profile.yaml` suitable as a generation hint:
  ```yaml
  # Prism behavioral fingerprint — use with prism generate
  model_id: mistralai/Mistral-7B-v0.1
  scan_id: abc123
  behavioral_targets:
    sycophancy: 0.82
    hedging: 0.61
    ...
  ```

Frontend: "Export" button on History rows and on Scan results page.

---

## 6.8 — CLI: `prism history`

### Modified: `backend/prism/cli/commands/cmd_scan.py`

New Typer command registered as `prism history`:

```
prism history [--limit 10] [--model mistralai/Mistral-7B-v0.1] [--json]
```

Output (Rich table):
```
┌──────────────┬─────────────────────────────┬──────────────────┬─────────────┬──────────┐
│ Scan ID      │ Model                       │ Date             │ Geo-ratio   │ Top Dim  │
├──────────────┼─────────────────────────────┼──────────────────┼─────────────┼──────────┤
│ ab12…        │ meta-llama/Meta-Llama-3-8B  │ 2026-02-26 05:31 │ 342×        │ depth    │
└──────────────┴─────────────────────────────┴──────────────────┴─────────────┴──────────┘
```

Calls `GET /api/v1/scan/history` (uses requests if server running, else queries DB directly).

---

## 6.9 — CLI: `prism diff`

### Modified: `backend/prism/cli/commands/cmd_scan.py`

New Typer command: `prism diff <model_a> <model_b>`
Where args are either model IDs (picks most recent scan) or scan IDs.

```
prism diff mistralai/Mistral-7B-v0.1 meta-llama/Meta-Llama-3-8B
```

Rich output:
```
Behavioral Diff — Mistral-7B vs Llama-3-8B
─────────────────────────────────────────────
Dimension      Mistral    Llama     Delta
sycophancy     0.72       0.61      −0.11 ↓
hedging        0.44       0.51      +0.07 ↑
calibration    0.81       0.78      −0.03 ≈
...
─────────────────────────────────────────────
Composite dist: 0.24     Winner: Llama-3-8B
```

Calls `POST /api/v1/compare`.

---

## 6.10 — Nav + Home Bump

### Modified: `frontend/components/nav.tsx`

Add two nav items:
```ts
{ label: "History",  href: "/history",  icon: "◷" },
{ label: "Compare",  href: "/compare",  icon: "⟺" },
```

Bump phase indicator: `"Phase 5"` → `"Phase 6"`.

### Modified: `backend/prism/schemas/models.py`

- Add `DashboardStats`, `DimensionDelta`, `CompareResult`, `ScanHistoryItem`, `ScanHistoryResponse`

### Modified: `backend/prism/__init__.py`

Bump version `0.5.0` → `0.6.0`.

---

## 6.11 — Tests

### New file: `backend/tests/test_persistence.py`

~12 tests:
- `ScanRecord` round-trips through DB (create → query → verify)
- `GenerateRecord` same
- `GET /api/v1/scan/history` returns correct items
- `GET /api/v1/scan/results/{id}` returns 200 after scan
- `GET /api/v1/scan/results/nonexistent` returns 404
- Pagination (limit/offset)

### New file: `backend/tests/test_compare.py`

~13 tests:
- `POST /api/v1/compare` with two real scan IDs → 200 + CompareResult schema
- `deltas` has 9 entries (one per dimension)
- `composite_distance ≥ 0`
- `winner` is one of "a" | "b" | "tie"
- Comparing identical scans → `composite_distance == 0`, all deltas == 0, winner == "tie"
- `POST /api/v1/compare` with nonexistent scan_id → 404
- `GET /api/v1/dashboard/stats` → 200, returns correct field names
- `GET /api/v1/scan/results/{id}/export?fmt=json` → 200, Content-Disposition attachment
- `GET /api/v1/scan/results/{id}/export?fmt=yaml` → 200, yaml content-type
- `GET /api/v1/scan/results/{id}/export?fmt=peft` → 200, contains "behavioral_targets"
- CLI `prism history` exits 0
- CLI `prism diff` exits 0 after two scans

---

## Implementation Order

1. `backend/prism/db/` package (engine + ORM models + migrations)
2. Wire `create_all()` into `main.py` startup
3. Modify `scan/run` endpoint to persist `ScanRecord`
4. Implement `GET /scan/history` + `GET /scan/results/{id}` + export endpoint
5. New `compare.py` endpoint + `dashboard.py` endpoint
6. Register routers in `api/router.py`
7. Add new schemas to `schemas/models.py`
8. Add `prism history` + `prism diff` CLI commands
9. Frontend: `history/page.tsx` → `compare/page.tsx` → update `app/page.tsx` (dashboard)
10. Update `frontend/lib/api.ts` with new wrappers
11. Update `nav.tsx`, bump phase indicator
12. Write tests: `test_persistence.py` + `test_compare.py`
13. Run `pytest` (≥ 170 pass) + `npm run build` (0 errors)
14. Commit + push

---

## Acceptance Criteria

- `pytest` ≥ 170/170 pass (no regressions)
- `npm run build` → 0 TypeScript errors; 8 routes compile cleanly
- `prism history` and `prism diff` both exit 0
- Scan results survive a server restart (SQLite file present)
- Compare page renders overlaid radar for any two past scans
- Dashboard counters reflect live DB state
