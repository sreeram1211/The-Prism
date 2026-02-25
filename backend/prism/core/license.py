"""
License verification — Phase 6 skeleton.

Supports two modes (set via LICENSE_MODE env var):
  - "online":  Verifies license key against the BuildMaxxing license server.
  - "offline": Validates a locally-signed license file (for air-gapped installs).

In Phase 1 the license system is a skeleton — all feature gates return True
in "offline" mode so the developer experience is frictionless during local dev.
The real enforcement is added in Phase 6 alongside Stripe billing.

License tiers:
  - Free:  Prism Scan only.
  - Pro:   Scan + Generate + Monitor + Agent  ($20/mo via Stripe).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import httpx

from prism.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# License tier
# ---------------------------------------------------------------------------

class LicenseTier(str, Enum):
    FREE = "free"
    PRO = "pro"


# ---------------------------------------------------------------------------
# License info
# ---------------------------------------------------------------------------

@dataclass
class LicenseInfo:
    valid: bool
    tier: LicenseTier
    user_id: str | None
    email: str | None
    expires_at: str | None  # ISO-8601 or None for perpetual
    message: str


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class LicenseVerifier:
    """
    Validates a Prism license key.

    Usage::

        verifier = LicenseVerifier()
        info = verifier.verify("PRISM-XXXX-XXXX-XXXX")
        if info.tier == LicenseTier.PRO:
            ...
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def verify(self, license_key: str) -> LicenseInfo:
        """
        Verify a license key.

        In Phase 1 (skeleton mode) this always returns a free-tier license
        when LICENSE_MODE=offline.  Phase 6 wires in the real network check.
        """
        if self._settings.license_mode == "offline":
            return self._verify_offline(license_key)
        return self._verify_online(license_key)

    def _verify_offline(self, license_key: str) -> LicenseInfo:
        """
        Local verification using a simple key checksum.

        Format:  PRISM-TIER-XXXXXXXX
          e.g.   PRISM-PRO-A1B2C3D4

        In Phase 1 we accept any key that starts with "PRISM-" as valid free
        and "PRISM-PRO-" as valid Pro.  Phase 6 replaces this with a real
        RSA-signed JWT embedded in the key blob.
        """
        key = license_key.strip().upper()

        if key.startswith("PRISM-PRO-"):
            return LicenseInfo(
                valid=True,
                tier=LicenseTier.PRO,
                user_id=_key_fingerprint(key),
                email=None,
                expires_at=None,
                message="Pro license verified (offline mode).",
            )

        if key.startswith("PRISM-"):
            return LicenseInfo(
                valid=True,
                tier=LicenseTier.FREE,
                user_id=_key_fingerprint(key),
                email=None,
                expires_at=None,
                message="Free license verified (offline mode).",
            )

        # Dev fallback: no key → treat as free
        return LicenseInfo(
            valid=True,
            tier=LicenseTier.FREE,
            user_id=None,
            email=None,
            expires_at=None,
            message="No license key provided — defaulting to Free tier (dev mode).",
        )

    def _verify_online(self, license_key: str) -> LicenseInfo:
        """
        Remote verification against the BuildMaxxing license server.

        Fully implemented in Phase 6.  Returns offline fallback if the
        server is unreachable (graceful degradation).
        """
        url = f"{self._settings.license_server_url}/v1/verify"
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(url, json={"key": license_key})
                resp.raise_for_status()
                data = resp.json()
                return LicenseInfo(
                    valid=data.get("valid", False),
                    tier=LicenseTier(data.get("tier", "free")),
                    user_id=data.get("user_id"),
                    email=data.get("email"),
                    expires_at=data.get("expires_at"),
                    message=data.get("message", ""),
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "License server unreachable (%s) — falling back to offline verification.", exc
            )
            return self._verify_offline(license_key)


def _key_fingerprint(key: str) -> str:
    """Derive a stable user_id from a license key for logging/tracking."""
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# Module-level singleton for convenience
_default_verifier: LicenseVerifier | None = None


def get_verifier() -> LicenseVerifier:
    """Return the module-level LicenseVerifier singleton."""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = LicenseVerifier()
    return _default_verifier
