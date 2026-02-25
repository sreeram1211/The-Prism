"""
Shared pytest fixtures and test utilities for The Prism test suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from prism.main import app


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client() -> TestClient:
    """Synchronous FastAPI test client (session-scoped for speed)."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Synthetic model configs (no network access)
# ---------------------------------------------------------------------------

LLAMA3_8B_CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
    "max_position_embeddings": 8192,
}

MISTRAL_7B_CONFIG: dict[str, Any] = {
    "model_type": "mistral",
    "architectures": ["MistralForCausalLM"],
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
}

MIXTRAL_8X7B_CONFIG: dict[str, Any] = {
    "model_type": "mixtral",
    "architectures": ["MixtralForCausalLM"],
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "vocab_size": 32000,
}

MAMBA_130M_CONFIG: dict[str, Any] = {
    "model_type": "mamba",
    "architectures": ["MambaForCausalLM"],
    "num_hidden_layers": 24,
    "hidden_size": 768,
    "intermediate_size": 1536,
    "state_size": 16,
    "expand": 2,
    "vocab_size": 50280,
}

FALCON_MAMBA_CONFIG: dict[str, Any] = {
    "model_type": "falcon_mamba",
    "architectures": ["FalconMambaForCausalLM"],
    "num_hidden_layers": 64,
    "hidden_size": 4096,
    "intermediate_size": 8192,
    "state_size": 16,
    "vocab_size": 65024,
}

JAMBA_CONFIG: dict[str, Any] = {
    "model_type": "jamba",
    "architectures": ["JambaForCausalLM"],
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "state_size": 16,
    "vocab_size": 65536,
}

T5_BASE_CONFIG: dict[str, Any] = {
    "model_type": "t5",
    "architectures": ["T5ForConditionalGeneration"],
    "num_hidden_layers": 12,
    "d_model": 768,
    "d_ff": 3072,
    "num_heads": 12,
    "vocab_size": 32128,
}

GPT2_CONFIG: dict[str, Any] = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "vocab_size": 50257,
}

UNKNOWN_CONFIG: dict[str, Any] = {
    "model_type": "totally_custom_arch",
    "architectures": ["CustomModel"],
    "num_hidden_layers": 16,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_attention_heads": 16,
    "vocab_size": 32000,
}

ALL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama3-8b": LLAMA3_8B_CONFIG,
    "mistral-7b": MISTRAL_7B_CONFIG,
    "mixtral-8x7b": MIXTRAL_8X7B_CONFIG,
    "mamba-130m": MAMBA_130M_CONFIG,
    "falcon-mamba": FALCON_MAMBA_CONFIG,
    "jamba": JAMBA_CONFIG,
    "t5-base": T5_BASE_CONFIG,
    "gpt2": GPT2_CONFIG,
    "unknown": UNKNOWN_CONFIG,
}


@pytest.fixture
def llama3_config() -> dict[str, Any]:
    return dict(LLAMA3_8B_CONFIG)


@pytest.fixture
def mamba_config() -> dict[str, Any]:
    return dict(MAMBA_130M_CONFIG)


@pytest.fixture
def mixtral_config() -> dict[str, Any]:
    return dict(MIXTRAL_8X7B_CONFIG)
