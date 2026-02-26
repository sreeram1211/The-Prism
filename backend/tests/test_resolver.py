"""
Tests for the Prism Auto-Resolver module.

Coverage:
  - Architecture detection for all supported model families
  - LoRA target mapping (full + minimal)
  - Parameter count estimation
  - GQA detection
  - MoE detection
  - Fallback for unknown architectures
  - PrismAutoResolver.resolve() with mocked HuggingFace downloads
  - PrismAutoResolver.resolve_from_config() (no network)
  - Error handling: model not found, gated repo, parse error
  - FastAPI endpoint integration tests (via TestClient)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from prism.resolver.arch_detector import ArchitectureFamily, detect_architecture
from prism.resolver.auto_resolver import (
    ConfigParseError,
    GatedModelError,
    ModelNotFoundError,
    PrismAutoResolver,
    ResolverError,
)
from prism.resolver.lora_targets import LORA_TARGET_MAP, get_lora_targets

from .conftest import (
    ALL_CONFIGS,
    FALCON_MAMBA_CONFIG,
    GPT2_CONFIG,
    JAMBA_CONFIG,
    LLAMA3_8B_CONFIG,
    MAMBA_130M_CONFIG,
    MISTRAL_7B_CONFIG,
    MIXTRAL_8X7B_CONFIG,
    T5_BASE_CONFIG,
    UNKNOWN_CONFIG,
)


# ===========================================================================
# arch_detector tests
# ===========================================================================

class TestArchitectureDetection:
    """Tests for detect_architecture() against various config dicts."""

    def test_llama_family(self):
        info = detect_architecture(LLAMA3_8B_CONFIG)
        assert info.model_type == "llama"
        assert info.family == ArchitectureFamily.ATTENTION
        assert info.num_hidden_layers == 32
        assert info.hidden_size == 4096
        assert info.num_attention_heads == 32
        assert info.num_key_value_heads == 8
        assert info.uses_gqa is True
        assert info.is_moe is False
        assert info.param_count_estimate > 0
        assert info.model_size_gb_bf16 > 0

    def test_mistral_family(self):
        info = detect_architecture(MISTRAL_7B_CONFIG)
        assert info.model_type == "mistral"
        assert info.family == ArchitectureFamily.ATTENTION
        assert info.uses_gqa is True  # 8 KV heads vs 32 attention heads

    def test_mixtral_is_moe(self):
        info = detect_architecture(MIXTRAL_8X7B_CONFIG)
        assert info.model_type == "mixtral"
        assert info.family == ArchitectureFamily.ATTENTION
        assert info.is_moe is True
        assert info.num_experts == 8
        assert info.num_experts_per_token == 2

    def test_mamba_is_ssm(self):
        info = detect_architecture(MAMBA_130M_CONFIG)
        assert info.model_type == "mamba"
        assert info.family == ArchitectureFamily.SSM
        assert info.num_attention_heads is None
        assert info.num_key_value_heads is None
        assert info.uses_gqa is False
        assert info.state_size == 16
        assert info.ssm_expansion_factor == 2.0

    def test_falcon_mamba_is_ssm(self):
        info = detect_architecture(FALCON_MAMBA_CONFIG)
        assert info.family == ArchitectureFamily.SSM
        assert info.num_attention_heads is None

    def test_jamba_is_hybrid(self):
        info = detect_architecture(JAMBA_CONFIG)
        assert info.model_type == "jamba"
        assert info.family == ArchitectureFamily.HYBRID
        # Hybrid: has both attention and SSM fields
        assert info.num_attention_heads is not None
        assert info.state_size is not None

    def test_t5_is_encoder_decoder(self):
        info = detect_architecture(T5_BASE_CONFIG)
        assert info.model_type == "t5"
        assert info.family == ArchitectureFamily.ENCODER_DECODER
        # T5 uses d_model for hidden_size
        assert info.hidden_size == 768

    def test_gpt2_parses_n_layer(self):
        info = detect_architecture(GPT2_CONFIG)
        assert info.model_type == "gpt2"
        assert info.family == ArchitectureFamily.ATTENTION
        # GPT-2 uses n_layer, n_embd
        assert info.num_hidden_layers == 12
        assert info.hidden_size == 768

    def test_unknown_architecture_defaults(self):
        info = detect_architecture(UNKNOWN_CONFIG)
        assert info.model_type == "totally_custom_arch"
        assert info.family == ArchitectureFamily.ATTENTION  # safe default
        assert info.num_hidden_layers == 16
        assert info.hidden_size == 2048

    def test_param_count_scales_with_size(self):
        """Larger models should have higher parameter estimates."""
        small = detect_architecture(MAMBA_130M_CONFIG)
        large = detect_architecture(LLAMA3_8B_CONFIG)
        assert large.param_count_estimate > small.param_count_estimate

    def test_lora_rank_scales_with_param_count(self):
        """LoRA rank recommendation should increase with model size."""
        small = detect_architecture(GPT2_CONFIG)
        large = detect_architecture(LLAMA3_8B_CONFIG)
        assert large.lora_rank_recommendation >= small.lora_rank_recommendation

    def test_lora_rank_recommendations(self):
        """Check the rank heuristic thresholds."""
        small_info = detect_architecture(MAMBA_130M_CONFIG)
        assert small_info.lora_rank_recommendation == 8  # < 1B → rank 8

    def test_model_size_gb_is_positive(self):
        for name, config in ALL_CONFIGS.items():
            info = detect_architecture(config)
            assert info.model_size_gb_bf16 > 0, f"size should be >0 for {name}"


# ===========================================================================
# lora_targets tests
# ===========================================================================

class TestLoraTargets:
    """Tests for get_lora_targets() and the LORA_TARGET_MAP."""

    def test_llama_full_targets(self):
        targets = get_lora_targets("llama")
        assert "q_proj" in targets
        assert "k_proj" in targets
        assert "v_proj" in targets
        assert "o_proj" in targets
        assert "gate_proj" in targets
        assert "up_proj" in targets
        assert "down_proj" in targets

    def test_llama_minimal_targets(self):
        targets = get_lora_targets("llama", minimal=True)
        assert targets == ["q_proj", "v_proj"]

    def test_mamba_targets_no_attention(self):
        targets = get_lora_targets("mamba")
        assert "q_proj" not in targets
        assert "in_proj" in targets
        assert "out_proj" in targets

    def test_jamba_hybrid_targets(self):
        targets = get_lora_targets("jamba")
        # Jamba has both attention AND SSM targets
        assert "q_proj" in targets
        assert "in_proj" in targets

    def test_t5_targets(self):
        targets = get_lora_targets("t5")
        assert "q" in targets
        assert "v" in targets
        assert "wi" in targets

    def test_unknown_type_fallback(self):
        targets = get_lora_targets("completely_unknown_model_xyz123")
        # Should fall back gracefully, not raise
        assert isinstance(targets, list)
        assert len(targets) > 0

    def test_case_insensitive_lookup(self):
        """model_type should be lowercased before lookup."""
        targets_lower = get_lora_targets("llama")
        targets_upper = get_lora_targets("LLAMA")
        assert targets_lower == targets_upper

    def test_prefix_match(self):
        """'llama3' should match the 'llama' key."""
        targets = get_lora_targets("llama3")
        assert "q_proj" in targets  # uses llama targets

    def test_all_targets_are_lists_of_strings(self):
        for model_type, targets in LORA_TARGET_MAP.items():
            assert isinstance(targets, list), f"{model_type} targets must be a list"
            for t in targets:
                assert isinstance(t, str), f"{model_type} target {t!r} must be a string"

    def test_no_duplicate_targets(self):
        for model_type, targets in LORA_TARGET_MAP.items():
            assert len(targets) == len(set(targets)), (
                f"{model_type} has duplicate LoRA targets"
            )


# ===========================================================================
# PrismAutoResolver tests (mocked HF)
# ===========================================================================

class TestPrismAutoResolver:
    """Tests for PrismAutoResolver with mocked HuggingFace downloads."""

    def _make_resolver(self, hf_token: str | None = None) -> PrismAutoResolver:
        return PrismAutoResolver(hf_token=hf_token)

    def _patch_hf_download(self, config: dict[str, Any]):
        """Context manager: patches hf_hub_download to return a temp file."""
        import contextlib

        @contextlib.contextmanager
        def _patcher():
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(config, f)
                tmpfile = f.name
            with patch("prism.resolver.auto_resolver.hf_hub_download", return_value=tmpfile):
                yield

        return _patcher()

    def test_resolve_llama(self):
        resolver = self._make_resolver()
        with self._patch_hf_download(LLAMA3_8B_CONFIG):
            result = resolver.resolve("meta-llama/Meta-Llama-3-8B")

        assert result.model_id == "meta-llama/Meta-Llama-3-8B"
        assert result.arch.model_type == "llama"
        assert result.arch.family == ArchitectureFamily.ATTENTION
        assert "q_proj" in result.lora_targets
        assert result.arch.lora_rank_recommendation == 32  # 8B → 7B-14B range

    def test_resolve_mamba(self):
        resolver = self._make_resolver()
        with self._patch_hf_download(MAMBA_130M_CONFIG):
            result = resolver.resolve("state-spaces/mamba-130m-hf")

        assert result.arch.family == ArchitectureFamily.SSM
        assert "in_proj" in result.lora_targets
        assert result.arch.lora_rank_recommendation == 8  # < 1B

    def test_resolve_mistral_gqa(self):
        resolver = self._make_resolver()
        with self._patch_hf_download(MISTRAL_7B_CONFIG):
            result = resolver.resolve("mistralai/Mistral-7B-v0.1")

        assert result.arch.uses_gqa is True

    def test_resolve_mixtral_moe(self):
        resolver = self._make_resolver()
        with self._patch_hf_download(MIXTRAL_8X7B_CONFIG):
            result = resolver.resolve("mistralai/Mixtral-8x7B-v0.1")

        assert result.arch.is_moe is True
        assert result.arch.num_experts == 8

    def test_to_dict_completeness(self):
        """to_dict() must include all expected keys."""
        resolver = self._make_resolver()
        with self._patch_hf_download(LLAMA3_8B_CONFIG):
            result = resolver.resolve("meta-llama/Meta-Llama-3-8B")

        d = result.to_dict()
        required_keys = [
            "model_id", "revision", "model_type", "architectures", "family",
            "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim", "uses_gqa",
            "is_moe", "num_experts", "num_experts_per_token",
            "state_size", "ssm_expansion_factor", "vocab_size",
            "param_count_estimate", "model_size_gb_bf16",
            "lora_rank_recommendation", "lora_targets", "lora_targets_minimal",
        ]
        for key in required_keys:
            assert key in d, f"Missing key in to_dict(): {key!r}"

    def test_config_cache_avoids_double_download(self):
        resolver = self._make_resolver()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(LLAMA3_8B_CONFIG, f)
            tmpfile = f.name

        with patch(
            "prism.resolver.auto_resolver.hf_hub_download", return_value=tmpfile
        ) as mock_dl:
            resolver.resolve("org/model-a")
            resolver.resolve("org/model-a")  # second call — should hit cache
            assert mock_dl.call_count == 1  # only downloaded once

    def test_resolve_from_config_no_network(self):
        """resolve_from_config() must work without any network calls."""
        resolver = self._make_resolver()
        result = resolver.resolve_from_config(JAMBA_CONFIG, model_id="local/jamba")

        assert result.model_id == "local/jamba"
        assert result.arch.family == ArchitectureFamily.HYBRID

    def test_model_not_found_raises(self):
        from huggingface_hub.utils import RepositoryNotFoundError as HFNotFound

        resolver = self._make_resolver()
        with patch(
            "prism.resolver.auto_resolver.hf_hub_download",
            side_effect=HFNotFound(message="not found", response=MagicMock()),
        ):
            with pytest.raises(ModelNotFoundError):
                resolver.resolve("definitely/does-not-exist-xyz")

    def test_gated_model_raises(self):
        from huggingface_hub.utils import GatedRepoError as HFGatedError

        resolver = self._make_resolver()
        with patch(
            "prism.resolver.auto_resolver.hf_hub_download",
            side_effect=HFGatedError(message="gated", response=MagicMock()),
        ):
            with pytest.raises(GatedModelError):
                resolver.resolve("meta-llama/Llama-2-7b-hf")

    def test_corrupt_config_raises(self):
        """A syntactically invalid config.json must raise ConfigParseError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("THIS IS NOT JSON !!!")
            tmpfile = f.name

        resolver = self._make_resolver()
        with patch("prism.resolver.auto_resolver.hf_hub_download", return_value=tmpfile):
            with pytest.raises(ConfigParseError):
                resolver.resolve("org/corrupt-model")

    def test_repr(self):
        resolver = self._make_resolver()
        with self._patch_hf_download(LLAMA3_8B_CONFIG):
            result = resolver.resolve("org/model")

        r = repr(result)
        assert "org/model" in r
        assert "attention" in r


# ===========================================================================
# FastAPI integration tests
# ===========================================================================

class TestResolverEndpoints:
    """Integration tests via FastAPI TestClient."""

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_root_redirect(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["product"] == "The Prism"
        assert "docs" in data

    def test_resolve_endpoint_success(self, client):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(MISTRAL_7B_CONFIG, f)
            tmpfile = f.name

        with patch("prism.resolver.auto_resolver.hf_hub_download", return_value=tmpfile):
            resp = client.post(
                "/api/v1/resolver/resolve",
                json={"model_id": "mistralai/Mistral-7B-v0.1"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "mistral"
        assert data["family"] == "attention"
        assert data["uses_gqa"] is True
        assert isinstance(data["lora_targets"], list)
        assert len(data["lora_targets"]) > 0

    def test_resolve_endpoint_model_not_found(self, client):
        from huggingface_hub.utils import RepositoryNotFoundError as HFNotFound

        with patch(
            "prism.resolver.auto_resolver.hf_hub_download",
            side_effect=HFNotFound(message="not found", response=MagicMock()),
        ):
            resp = client.post(
                "/api/v1/resolver/resolve",
                json={"model_id": "no-one/does-not-exist"},
            )

        assert resp.status_code == 404

    def test_resolve_endpoint_gated(self, client):
        from huggingface_hub.utils import GatedRepoError as HFGatedError

        with patch(
            "prism.resolver.auto_resolver.hf_hub_download",
            side_effect=HFGatedError(message="gated", response=MagicMock()),
        ):
            resp = client.post(
                "/api/v1/resolver/resolve",
                json={"model_id": "meta-llama/Llama-2-7b-hf"},
            )

        assert resp.status_code == 403

    def test_resolve_endpoint_invalid_model_id(self, client):
        resp = client.post(
            "/api/v1/resolver/resolve",
            json={"model_id": ""},
        )
        assert resp.status_code == 422  # Pydantic validation failure

    def test_get_model_info_endpoint(self, client):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(LLAMA3_8B_CONFIG, f)
            tmpfile = f.name

        with patch("prism.resolver.auto_resolver.hf_hub_download", return_value=tmpfile):
            resp = client.get("/api/v1/resolver/info/meta-llama/Meta-Llama-3-8B")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "llama"

    def test_scan_run_returns_scan_result(self, client):
        resp = client.post(
            "/api/v1/scan/run",
            json={"model_id": "mistralai/Mistral-7B-v0.1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "mistralai/Mistral-7B-v0.1"
        assert len(data["scores"]) == 9
        assert "geometric_separation_ratio" in data
        assert "scan_duration_ms" in data
        # Verify score range
        for s in data["scores"]:
            assert 0.0 <= s["score"] <= 1.0

    def test_generate_endpoint_live(self, client):
        # Phase 4 is now implemented — missing body returns 422 (validation)
        resp = client.post("/api/v1/generate/lora")
        assert resp.status_code == 422

    def test_monitor_endpoint_live(self, client):
        # Phase 5 is now implemented — missing body returns 422 (validation)
        resp = client.post("/api/v1/monitor/sessions")
        assert resp.status_code == 422

    def test_agent_endpoint_live(self, client):
        # Phase 5 is now implemented — missing body returns 422 (validation)
        resp = client.post("/api/v1/agent/chat")
        assert resp.status_code == 422

    def test_openapi_schema_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "The Prism API"
