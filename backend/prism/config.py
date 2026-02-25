"""
Application-wide settings loaded from environment variables / .env file.
All values can be overridden via environment; defaults are safe for local dev.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "The Prism"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── API ──────────────────────────────────────────────────────────────────
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # ── HuggingFace ──────────────────────────────────────────────────────────
    hf_token: str | None = None
    hf_cache_dir: str = "~/.prism/models"

    # ── Auth / JWT ────────────────────────────────────────────────────────────
    jwt_secret_key: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7  # 7 days

    # ── License ───────────────────────────────────────────────────────────────
    license_server_url: str = "https://license.buildmaxxing.com"
    license_mode: Literal["online", "offline"] = "online"

    # ── Stripe (Phase 6) ─────────────────────────────────────────────────────
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_pro_price_id: str | None = None

    # ── Vector DB (Phase 5) ──────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    vector_backend: Literal["qdrant", "chroma"] = "qdrant"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()
