# The Prism

**Local-first AI behavioral manifold tooling suite by BuildMaxxing.**

The Prism maps, monitors, and manipulates LLM behavior using a proprietary
16-dimensional fiber space. It gives you a full diagnostic and steering pipeline
for any HuggingFace model — from architecture detection through behavioral
scanning, LoRA config synthesis, live training telemetry, and an autonomous
agent — all running entirely on your own machine without sending weights
anywhere.

---

## What We Are Building

Most LLM tooling treats a model as a black box: you send prompts and get text
back. The Prism treats it as a **geometric object** — a point in a
high-dimensional behavioral manifold — and gives you instruments to read that
geometry and reshape it.

The full pipeline has six modules:

| Module | What it does |
|--------|-------------|
| **Auto-Resolver** | Downloads only `config.json` from HuggingFace Hub and reconstructs the full architecture descriptor: layer count, hidden size, attention geometry, MoE/SSM flags, parameter estimate, and ready-to-use LoRA target lists. |
| **Behavioral Scan** | Probes the model across 9 behavioral dimensions (sycophancy, hedging, calibration, depth, coherence, focus, specificity, verbosity, repetition) and returns dimension scores plus a 16-dimensional geometric separation ratio derived from the fiber space projection. |
| **LoRA Generator** | Accepts 9 behavioral slider values (0.0–1.0) and derives optimal LoRA hyperparameters (rank, alpha, dropout, target modules) via fiber-space heuristics. Outputs a PEFT-compatible `adapter_config.json` and a ready-to-run training YAML. |
| **Live Monitor** | Streams real-time training telemetry — loss curves, gradient norms, adapter convergence diagnostics — over WebSocket to a live dashboard. |
| **Agent** | Autonomous chat with keyword-based vector memory (dense embeddings via Qdrant planned for Phase 6) and RSI α' improvement-acceleration tracking across turns. |
| **Persistence + Comparison** *(Phase 6)* | SQLite-backed history of all scans and jobs; side-by-side radar overlay for model comparison; behavioral diff; export to JSON/YAML/PEFT. |

---

## Current State — Phase 5 Complete

### Backend (FastAPI + Python)

**Auto-Resolver** — `POST /api/v1/resolver/resolve`
Calls HuggingFace Hub metadata API (no weights). Detects model family
(attention / SSM / hybrid / encoder-decoder), GQA, MoE, SSM expansion factor,
parameter count, and bf16 size estimate. Returns LoRA rank recommendation and
full/minimal target module lists.

**Behavioral Scan** — `POST /api/v1/scan/run`
Mock engine seeded from the model ID (deterministic, architecture-family-biased
score distributions). Returns 9 `DimensionScore` objects with human-readable
interpretations and a geometric separation ratio (125×–1376× range). Real probe
engine is Phase 2+ milestone.

**LoRA Generator** — `POST /api/v1/generate/lora`
Derives rank (log-scale, complexity-weighted, snapped to power-of-2), alpha
(2×rank + assertiveness boost), dropout (coherence/focus-weighted), and target
modules. Emits PEFT `adapter_config.json` and LLaMA-Factory/Axolotl YAML.

**Live Monitor** — `POST /api/v1/monitor/sessions` + `WS /api/v1/monitor/ws/{id}`
Creates a telemetry session then streams synthetic activation frames over
WebSocket at ~2 Hz. Real activation hooking is a Phase 6 milestone.

**Agent** — `POST /api/v1/agent/chat`
Session-scoped conversational agent with in-process memory store, keyword
overlap retrieval (TF-IDF style with regex tokenization), and RSI α' metric
(second derivative of per-turn quality signal). Sessions are server-generated
— client-supplied IDs are only reused if the session already exists.

### CLI (`prism` command)

```
prism resolve <model_id>       # Architecture detection + LoRA recommendations
prism scan    <model_id>       # 9-dimensional behavioral diagnostic
prism generate <model_id>      # Compile a LoRA adapter config
prism monitor  <model_id>      # Activation telemetry stream
prism agent                    # Autonomous agent chat
prism serve                    # Start the FastAPI server
prism info                     # System info + dependency versions
```

All CLI commands render with Rich — colour-coded tables, progress bars, and
dimension radar output.

### Frontend (Next.js 16 + Tailwind)

| Route | Page |
|-------|------|
| `/` | Landing page with feature cards |
| `/resolve` | Auto-Resolver form + architecture card |
| `/scan` | Scan form + interactive radar chart + heatmap |
| `/generate` | Behavioral sliders → LoRA config + training YAML |
| `/monitor` | WebSocket live telemetry dashboard |
| `/agent` | Chat interface with memory hit badges + α' indicator |

---

## Phase Roadmap

```
Phase 1  ✅  Auto-Resolver — architecture detection, LoRA target recommendations
Phase 2  ✅  CLI (Typer/Rich) + mock scan engine
Phase 3  ✅  Next.js 16 frontend — Resolve + Scan UI, radar chart, heatmap
Phase 4  ✅  LoRA Generator — behavioral sliders → PEFT adapter config + YAML
Phase 5  ✅  Live Monitor (WebSocket telemetry) + Agent (memory + α')
Phase 6  🔲  Persistence + Comparison — SQLite history, model diff, export
Phase 7  🔲  Real behavioral probe engine (C++/Python, ROC-AUC verified)
Phase 8  🔲  Dense vector memory (Qdrant), semantic clustering
Phase 9  🔲  Real activation hooking for live monitor
```

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `test_resolver.py` | 38 | ✅ pass |
| `test_scan_engine.py` | 14 | ✅ pass |
| `test_generate_engine.py` | 28 | ✅ pass |
| `test_agent.py` | 40 | ✅ pass |
| `test_cli.py` | 25 | ✅ pass |
| **Total** | **145** | **145 / 145** |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend framework | FastAPI 0.115 + Uvicorn |
| Validation | Pydantic v2 |
| CLI | Typer + Rich |
| HuggingFace integration | `huggingface-hub` (config-only, no weights) |
| Frontend framework | Next.js 16 (App Router) |
| Frontend language | TypeScript 5 |
| Styling | Tailwind CSS v4 |
| Charts | Recharts |
| UI primitives | Radix UI |
| Containerisation | Docker + Docker Compose |
| Test runner | pytest 8 |
| Linter / formatter | Ruff |

---

## Quick Start

### Backend

```bash
# Install dependencies
make install-dev

# Run tests
make test

# Start dev server (http://localhost:8000)
make dev
```

### Frontend

```bash
# Install dependencies
make frontend-install

# Start dev server (http://localhost:3000)
make frontend-dev

# Production build
make frontend-build
```

### Docker

```bash
make docker-up    # starts prism-api on :8000
make docker-down
```

### CLI

```bash
pip install -e backend/
prism info
prism resolve mistralai/Mistral-7B-Instruct-v0.2
prism scan    mistralai/Mistral-7B-Instruct-v0.2
```

---

## API Reference

Interactive docs at `http://localhost:8000/docs` (Swagger UI) or
`http://localhost:8000/redoc` once the server is running.

---

## Project Structure

```
The-Prism/
├── backend/
│   ├── prism/
│   │   ├── api/endpoints/      # FastAPI route handlers
│   │   ├── agent/              # Memory store + α' tracking
│   │   ├── cli/                # Typer commands + Rich renderers
│   │   ├── generate/           # LoRA hyperparameter engine
│   │   ├── monitor/            # WebSocket telemetry stream
│   │   ├── resolver/           # Architecture detection
│   │   ├── scan/               # Behavioral scan engine
│   │   └── schemas/            # Pydantic request/response models
│   └── tests/
├── frontend/
│   ├── app/                    # Next.js App Router pages
│   ├── components/             # Radar chart, heatmap, UI primitives
│   └── lib/                    # API client helpers
├── docker-compose.yml
└── Makefile
```

---

**BuildMaxxing © 2024-2026 — Proprietary**
