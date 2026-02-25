"""
JWT authentication utilities — Phase 6 skeleton.

Full implementation (Stripe-backed, license-gated) arrives in Phase 6.
This module provides the token creation/verification primitives so that
the rest of the codebase can reference them from day one.

Token claims structure:
    {
        "sub":   "<user_id>",
        "email": "<email>",
        "tier":  "free" | "pro",
        "exp":   <unix timestamp>
    }
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import BaseModel

from prism.config import Settings, get_settings

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Token models
# ---------------------------------------------------------------------------

class TokenPayload(BaseModel):
    sub: str
    email: str | None = None
    tier: str = "free"  # "free" | "pro"
    exp: int | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def create_access_token(
    user_id: str,
    email: str | None = None,
    tier: str = "free",
    extra_claims: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        user_id: Unique identifier for the user (e.g. Stripe customer ID).
        email: User's email address.
        tier: Subscription tier ("free" or "pro").
        extra_claims: Additional claims to embed in the token.
        settings: Application settings (uses singleton if None).

    Returns:
        Signed JWT string.
    """
    cfg = settings or get_settings()
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=cfg.jwt_expire_minutes)

    claims: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "tier": tier,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        **(extra_claims or {}),
    }

    return jwt.encode(claims, cfg.jwt_secret_key, algorithm=cfg.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> TokenPayload:
    """
    Decode and verify a JWT token.

    Args:
        token: Raw JWT string.
        settings: Application settings.

    Returns:
        Parsed TokenPayload.

    Raises:
        HTTPException 401 if token is invalid or expired.
    """
    cfg = settings or get_settings()
    try:
        payload = jwt.decode(token, cfg.jwt_secret_key, algorithms=[cfg.jwt_algorithm])
        return TokenPayload(**payload)
    except ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> TokenPayload:
    """
    FastAPI dependency: parse Bearer token and return the token payload.

    Usage::

        @router.get("/protected")
        def protected(user: TokenPayload = Depends(get_current_user)):
            ...

    Returns HTTP 401 if no valid Bearer token is present.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials, settings)


def require_pro(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """
    FastAPI dependency: ensures the authenticated user has a Pro license.

    Features gated behind this dependency:
      - Prism Generate (Phase 3)
      - Prism Monitor (Phase 4)
      - Prism Agent (Phase 5)

    Returns HTTP 403 if the user is on the free tier.
    """
    if user.tier != "pro":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "This feature requires a Prism Pro license ($20/mo). "
                "Upgrade at https://buildmaxxing.com/prism/upgrade"
            ),
        )
    return user
