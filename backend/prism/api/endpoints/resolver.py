"""
Prism Auto-Resolver API endpoints.

POST /api/v1/resolver/resolve  — resolve any HF model by ID
GET  /api/v1/resolver/info/{model_id}  — resolve with model_id as path param
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from prism.config import Settings, get_settings
from prism.resolver.auto_resolver import (
    GatedModelError,
    ModelNotFoundError,
    ConfigParseError,
    PrismAutoResolver,
    ResolverError,
)
from prism.schemas.models import ModelResolverResponse, ResolveRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resolver", tags=["Auto-Resolver"])


def _get_resolver(settings: Settings = Depends(get_settings)) -> PrismAutoResolver:
    """Dependency: returns a configured PrismAutoResolver instance."""
    return PrismAutoResolver(
        hf_token=settings.hf_token,
        cache_dir=settings.hf_cache_dir,
    )


def _run_resolve(
    model_id: str,
    revision: str,
    hf_token_override: str | None,
    resolver: PrismAutoResolver,
    settings: Settings,
) -> ModelResolverResponse:
    """Shared logic used by both POST and GET endpoints."""
    # Allow per-request token override (e.g. user supplies their own HF token)
    if hf_token_override:
        resolver = PrismAutoResolver(
            hf_token=hf_token_override,
            cache_dir=settings.hf_cache_dir,
        )

    try:
        result = resolver.resolve(model_id, revision=revision)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "model_not_found", "message": str(exc)},
        ) from exc
    except GatedModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "gated_model", "message": str(exc)},
        ) from exc
    except ConfigParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "config_parse_error", "message": str(exc)},
        ) from exc
    except ResolverError as exc:
        logger.exception("Unexpected resolver error for %s", model_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "resolver_error", "message": str(exc)},
        ) from exc

    return ModelResolverResponse(**result.to_dict())


@router.post(
    "/resolve",
    response_model=ModelResolverResponse,
    summary="Resolve a HuggingFace model",
    description=(
        "Downloads only config.json from HuggingFace Hub (no weights) and returns "
        "the complete architecture descriptor with recommended LoRA target modules."
    ),
    status_code=status.HTTP_200_OK,
)
def resolve_model(
    body: ResolveRequest,
    resolver: PrismAutoResolver = Depends(_get_resolver),
    settings: Settings = Depends(get_settings),
) -> ModelResolverResponse:
    return _run_resolve(
        model_id=body.model_id,
        revision=body.revision,
        hf_token_override=body.hf_token,
        resolver=resolver,
        settings=settings,
    )


@router.get(
    "/info/{model_id:path}",
    response_model=ModelResolverResponse,
    summary="Resolve a model via GET",
    description=(
        "Convenience GET endpoint — model_id is URL-encoded in the path. "
        "Use POST /resolve for full options including custom HF tokens."
    ),
    status_code=status.HTTP_200_OK,
)
def get_model_info(
    model_id: str,
    revision: str = "main",
    resolver: PrismAutoResolver = Depends(_get_resolver),
    settings: Settings = Depends(get_settings),
) -> ModelResolverResponse:
    return _run_resolve(
        model_id=model_id,
        revision=revision,
        hf_token_override=None,
        resolver=resolver,
        settings=settings,
    )
