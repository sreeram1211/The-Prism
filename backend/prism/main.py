"""
The Prism — FastAPI application entry point.

Run locally:
    uvicorn prism.main:app --reload --host 0.0.0.0 --port 8000

Or via Makefile:
    make dev
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from prism import __version__
from prism.api.router import api_router
from prism.config import get_settings
from prism.schemas.models import HealthResponse

logger = logging.getLogger(__name__)

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info(
        "🔷 The Prism %s starting up — phase 1 (Auto-Resolver active)",
        __version__,
    )
    yield
    logger.info("🔷 The Prism shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="The Prism API",
        description=(
            "Local-first AI behavioral manifold tooling suite by BuildMaxxing. "
            "Maps, monitors, and manipulates LLM behavior using a proprietary "
            "16-dimensional fiber space."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    application.include_router(api_router, prefix=settings.api_prefix)

    # ── Health check (outside /api/v1 for load-balancer probes) ──────────────
    @application.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    # ── Root redirect to docs ─────────────────────────────────────────────────
    @application.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "product": "The Prism",
                "version": __version__,
                "by": "BuildMaxxing",
                "docs": "/docs",
                "health": "/health",
            }
        )

    return application


app = create_app()
