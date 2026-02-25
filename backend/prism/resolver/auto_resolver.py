"""
Prism Auto-Resolver: downloads only config.json from HuggingFace Hub and
returns a complete ArchitectureInfo + LoRA recommendation without loading
any model weights.

Design principles:
- Strictly local-first: no telemetry to HF beyond the config download.
- Lightweight: the entire resolution process takes < 1 second on cold start.
- Deterministic: given the same model_id + revision, always returns the same result.
- Graceful: unknown architectures receive a documented fallback rather than an error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)

from .arch_detector import ArchitectureInfo, detect_architecture
from .lora_targets import get_lora_targets

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ResolverError(Exception):
    """Base class for Auto-Resolver errors."""


class ModelNotFoundError(ResolverError):
    """Raised when the model ID does not exist on HuggingFace Hub."""


class GatedModelError(ResolverError):
    """Raised when the model requires an HF token / access request."""


class ConfigParseError(ResolverError):
    """Raised when config.json is present but cannot be parsed."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ModelResolverResult:
    """
    Full resolution result for a HuggingFace model.

    This is a plain dataclass-like object (not Pydantic) so it can be used
    outside FastAPI contexts without importing Pydantic.  The API layer wraps
    this into a Pydantic response schema.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        arch: ArchitectureInfo,
        lora_targets: list[str],
        lora_targets_minimal: list[str],
        raw_config: dict[str, Any],
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.arch = arch
        self.lora_targets = lora_targets
        self.lora_targets_minimal = lora_targets_minimal
        self.raw_config = raw_config

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON responses."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "model_type": self.arch.model_type,
            "architectures": self.arch.architectures,
            "family": self.arch.family.value,
            "num_hidden_layers": self.arch.num_hidden_layers,
            "hidden_size": self.arch.hidden_size,
            "intermediate_size": self.arch.intermediate_size,
            "num_attention_heads": self.arch.num_attention_heads,
            "num_key_value_heads": self.arch.num_key_value_heads,
            "head_dim": self.arch.head_dim,
            "uses_gqa": self.arch.uses_gqa,
            "is_moe": self.arch.is_moe,
            "num_experts": self.arch.num_experts,
            "num_experts_per_token": self.arch.num_experts_per_token,
            "state_size": self.arch.state_size,
            "ssm_expansion_factor": self.arch.ssm_expansion_factor,
            "vocab_size": self.arch.vocab_size,
            "param_count_estimate": self.arch.param_count_estimate,
            "model_size_gb_bf16": self.arch.model_size_gb_bf16,
            "lora_rank_recommendation": self.arch.lora_rank_recommendation,
            "lora_targets": self.lora_targets,
            "lora_targets_minimal": self.lora_targets_minimal,
        }

    def __repr__(self) -> str:
        return (
            f"ModelResolverResult(model_id={self.model_id!r}, "
            f"family={self.arch.family.value}, "
            f"layers={self.arch.num_hidden_layers}, "
            f"params≈{self.arch.param_count_estimate / 1e9:.1f}B)"
        )


# ---------------------------------------------------------------------------
# Auto-Resolver
# ---------------------------------------------------------------------------

class PrismAutoResolver:
    """
    Resolves any HuggingFace model ID to a full architecture descriptor and
    LoRA target recommendation.

    Usage::

        resolver = PrismAutoResolver(hf_token="hf_xxx")
        result = resolver.resolve("meta-llama/Meta-Llama-3-8B")
        print(result.arch.family)         # ArchitectureFamily.ATTENTION
        print(result.lora_targets)        # ["q_proj", "k_proj", ...]
        print(result.arch.lora_rank_recommendation)  # 16
    """

    def __init__(
        self,
        hf_token: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            hf_token: Optional HuggingFace token for gated models.
            cache_dir: Local directory to cache downloaded config files.
                       Defaults to ~/.cache/huggingface/hub (HF default).
        """
        self.hf_token = hf_token
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._config_cache: dict[str, dict[str, Any]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        model_id: str,
        *,
        revision: str = "main",
    ) -> ModelResolverResult:
        """
        Resolve a HuggingFace model ID to a full ArchitectureInfo + LoRA targets.

        Only ``config.json`` is downloaded — no model weights are fetched.

        Args:
            model_id: HuggingFace model repository ID, e.g.
                      ``"meta-llama/Meta-Llama-3-8B"`` or
                      ``"mistralai/Mistral-7B-Instruct-v0.2"``.
            revision: Git revision / branch / tag to use (default: ``"main"``).

        Returns:
            ModelResolverResult containing the full architecture descriptor
            and LoRA target recommendations.

        Raises:
            ModelNotFoundError: Model ID not found on HF Hub.
            GatedModelError: Model is gated and requires an HF token.
            ConfigParseError: config.json exists but cannot be parsed.
            ResolverError: Any other resolution failure.
        """
        logger.info("Resolving model: %s @ %s", model_id, revision)

        config = self._fetch_config(model_id, revision)
        arch = detect_architecture(config)

        lora_targets = get_lora_targets(arch.model_type, minimal=False)
        lora_targets_minimal = get_lora_targets(arch.model_type, minimal=True)

        result = ModelResolverResult(
            model_id=model_id,
            revision=revision,
            arch=arch,
            lora_targets=lora_targets,
            lora_targets_minimal=lora_targets_minimal,
            raw_config=config,
        )

        logger.info(
            "Resolved %s → family=%s layers=%d params≈%.1fB lora_rank=%d",
            model_id,
            arch.family.value,
            arch.num_hidden_layers,
            arch.param_count_estimate / 1e9,
            arch.lora_rank_recommendation,
        )

        return result

    def resolve_from_config(
        self,
        config: dict[str, Any],
        model_id: str = "local-model",
        revision: str = "local",
    ) -> ModelResolverResult:
        """
        Resolve from a pre-loaded config dict (e.g. from a local model directory).

        Useful when the model is air-gapped and not accessible from HF Hub.

        Args:
            config: Parsed config.json dict.
            model_id: Identifier string (for display/logging only).
            revision: Revision string (for display only).
        """
        arch = detect_architecture(config)
        lora_targets = get_lora_targets(arch.model_type, minimal=False)
        lora_targets_minimal = get_lora_targets(arch.model_type, minimal=True)
        return ModelResolverResult(
            model_id=model_id,
            revision=revision,
            arch=arch,
            lora_targets=lora_targets,
            lora_targets_minimal=lora_targets_minimal,
            raw_config=config,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_config(self, model_id: str, revision: str) -> dict[str, Any]:
        """
        Download (or return cached) config.json for the given model.

        Uses the HuggingFace Hub to download only the config file, which is
        typically a few KB.  The downloaded file is cached locally by HF.

        Raises:
            ModelNotFoundError, GatedModelError, ConfigParseError, ResolverError
        """
        cache_key = f"{model_id}@{revision}"
        if cache_key in self._config_cache:
            logger.debug("Config cache hit for %s", cache_key)
            return self._config_cache[cache_key]

        try:
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                token=self.hf_token,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
            )
        except GatedRepoError as exc:
            # Must come before RepositoryNotFoundError — GatedRepoError is a
            # subclass of RepositoryNotFoundError in recent huggingface_hub versions.
            raise GatedModelError(
                f"Model '{model_id}' is gated. Provide an HF_TOKEN with "
                "access to this repository."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found on HuggingFace Hub. "
                "Check the model ID and your network connection."
            ) from exc
        except EntryNotFoundError as exc:
            raise ConfigParseError(
                f"config.json not found in '{model_id}'. "
                "This may not be a standard HuggingFace model repository."
            ) from exc
        except Exception as exc:
            raise ResolverError(
                f"Failed to download config for '{model_id}': {exc}"
            ) from exc

        try:
            with open(config_path, encoding="utf-8") as fh:
                config = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigParseError(
                f"Failed to parse config.json for '{model_id}': {exc}"
            ) from exc

        self._config_cache[cache_key] = config
        return config
