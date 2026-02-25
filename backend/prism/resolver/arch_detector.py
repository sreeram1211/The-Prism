"""
Architecture detection from a HuggingFace model config dict.

Parses the raw JSON from config.json (no model weights required) and returns
a structured ArchitectureInfo object used by the Auto-Resolver and LoRA
target selector.

Supports:
- Standard multi-head / grouped-query attention (LLaMA, Mistral, Qwen2, …)
- SSM / state-space models (Mamba, Falcon-Mamba)
- Hybrid attention+SSM (Jamba, Zamba)
- Encoder-decoder (T5, BART)
- GPT-style (GPT-2, GPT-NeoX, BLOOM, Falcon)
- MoE variants (Mixtral, Qwen2-MoE)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Architecture family classification
# ---------------------------------------------------------------------------

class ArchitectureFamily(str, Enum):
    """High-level family used by the Prism engine for manifold geometry."""

    ATTENTION = "attention"          # Standard transformer with attention layers
    SSM = "ssm"                      # Pure state-space model (no attention)
    HYBRID = "hybrid"                # Mamba + Attention layers interleaved
    ENCODER_DECODER = "encoder_decoder"  # Seq2seq (T5, BART, …)
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Structured architecture descriptor
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureInfo:
    """All architecture-relevant metadata extracted from a model config."""

    # Core identity
    model_type: str
    architectures: list[str]
    family: ArchitectureFamily

    # Layer geometry
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int

    # Attention geometry (None for pure SSMs)
    num_attention_heads: int | None
    num_key_value_heads: int | None  # < num_attention_heads → GQA
    head_dim: int | None             # per-head dimension

    # MoE specifics
    is_moe: bool
    num_experts: int | None
    num_experts_per_token: int | None

    # SSM specifics
    state_size: int | None           # d_state for Mamba-style layers
    ssm_expansion_factor: float | None

    # Vocabulary
    vocab_size: int

    # Derived estimates
    param_count_estimate: int        # approximate total parameters
    model_size_gb_bf16: float        # assuming bfloat16 storage

    # Raw pass-through for anything not explicitly parsed
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def uses_gqa(self) -> bool:
        """True when the model uses Grouped-Query Attention."""
        if self.num_attention_heads is None or self.num_key_value_heads is None:
            return False
        return self.num_key_value_heads < self.num_attention_heads

    @property
    def lora_rank_recommendation(self) -> int:
        """
        Heuristic LoRA rank based on model size and family.

        Larger models benefit from higher rank to capture enough behavioral
        signal; SSMs with smaller state spaces can use lower rank.
        """
        if self.param_count_estimate < 1e9:       # < 1B
            return 8
        elif self.param_count_estimate < 7e9:     # 1B – 7B
            return 16
        elif self.param_count_estimate < 14e9:    # 7B – 14B
            return 32
        elif self.param_count_estimate < 35e9:    # 14B – 35B
            return 64
        else:                                      # > 35B
            return 128


# ---------------------------------------------------------------------------
# Architecture type → family mapping
# ---------------------------------------------------------------------------

_SSM_TYPES: frozenset[str] = frozenset({
    "mamba", "mamba2", "falcon_mamba",
})

_HYBRID_TYPES: frozenset[str] = frozenset({
    "jamba", "zamba", "zamba2",
})

_ENCODER_DECODER_TYPES: frozenset[str] = frozenset({
    "t5", "mt5", "bart", "mbart", "pegasus", "flan-t5", "longt5",
    "led", "marian", "opus-mt",
})

_MoE_TYPES: frozenset[str] = frozenset({
    "mixtral", "qwen2_moe", "deepseek_v2", "deepseek_v3",
    "switch_transformers", "nllb-moe",
})

# Architecture class name suffixes → family override (for older configs that
# set model_type generically but have specific architecture class names).
_ARCH_CLASS_HINTS: dict[str, ArchitectureFamily] = {
    "MambaForCausalLM": ArchitectureFamily.SSM,
    "Mamba2ForCausalLM": ArchitectureFamily.SSM,
    "FalconMambaForCausalLM": ArchitectureFamily.SSM,
    "JambaForCausalLM": ArchitectureFamily.HYBRID,
    "T5ForConditionalGeneration": ArchitectureFamily.ENCODER_DECODER,
    "BartForConditionalGeneration": ArchitectureFamily.ENCODER_DECODER,
}


def _classify_family(model_type: str, architectures: list[str]) -> ArchitectureFamily:
    mt = model_type.lower()

    # Check architecture class names first (more specific)
    for arch_class in architectures:
        if arch_class in _ARCH_CLASS_HINTS:
            return _ARCH_CLASS_HINTS[arch_class]

    if mt in _SSM_TYPES:
        return ArchitectureFamily.SSM
    if mt in _HYBRID_TYPES:
        return ArchitectureFamily.HYBRID
    if mt in _ENCODER_DECODER_TYPES:
        return ArchitectureFamily.ENCODER_DECODER

    # Standard attention transformer (the common case)
    return ArchitectureFamily.ATTENTION


def _estimate_params(
    num_hidden_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
    vocab_size: int,
    num_experts: int | None,
    num_experts_per_token: int | None,
    is_ssm: bool,
) -> int:
    """
    Rough parameter count estimate derived from architecture dimensions.

    This is a heuristic for display/recommendation purposes only.
    Formula based on typical transformer layer structure.
    """
    if is_ssm:
        # Mamba-style: ~4 * L * D^2 for SSM projections + embeddings
        layer_params = 4 * hidden_size * hidden_size
        total = num_hidden_layers * layer_params + 2 * vocab_size * hidden_size
        return int(total)

    # Attention: Q, K, V, O projections
    if num_attention_heads and num_key_value_heads:
        head_dim = hidden_size // num_attention_heads
        q_size = hidden_size * hidden_size
        kv_size = 2 * num_key_value_heads * head_dim * hidden_size
        o_size = hidden_size * hidden_size
        attn_params = q_size + kv_size + o_size
    else:
        attn_params = 4 * hidden_size * hidden_size

    # FFN (MoE: multiply by num_experts)
    ffn_params = 3 * hidden_size * intermediate_size  # gate + up + down
    if num_experts and num_experts_per_token:
        # Only `num_experts` contribute params but not all are active per token
        ffn_params = ffn_params * num_experts

    # Layer norm: ~2 * D per layer
    ln_params = 2 * hidden_size

    layer_params = attn_params + ffn_params + ln_params
    total = num_hidden_layers * layer_params + 2 * vocab_size * hidden_size
    return int(total)


def detect_architecture(config: dict[str, Any]) -> ArchitectureInfo:
    """
    Parse a raw HuggingFace config.json dict and return an ArchitectureInfo.

    Args:
        config: Parsed JSON from config.json — no model weights needed.

    Returns:
        ArchitectureInfo with all detected fields populated.
    """
    model_type: str = config.get("model_type", "unknown")
    architectures: list[str] = config.get("architectures", [])

    family = _classify_family(model_type, architectures)
    is_ssm = family == ArchitectureFamily.SSM
    is_moe = model_type.lower() in _MoE_TYPES

    # ── Layer geometry ────────────────────────────────────────────────────────
    num_hidden_layers: int = (
        config.get("num_hidden_layers")
        or config.get("n_layer")        # GPT-2 style
        or config.get("n_layers")
        or config.get("num_layers")
        or 32                            # safe default
    )
    hidden_size: int = (
        config.get("hidden_size")
        or config.get("d_model")         # T5 / BART style
        or config.get("n_embd")          # GPT-2 style
        or 4096
    )
    intermediate_size: int = (
        config.get("intermediate_size")
        or config.get("d_ff")
        or config.get("ffn_dim")
        or config.get("inner_size")
        or int(hidden_size * 2.67)       # ~8/3 · D for SwiGLU-based FFNs
    )

    # ── Attention geometry ────────────────────────────────────────────────────
    num_attention_heads: int | None = (
        config.get("num_attention_heads")
        or config.get("n_head")
        or config.get("num_heads")
    ) if not is_ssm else None

    num_key_value_heads: int | None = (
        config.get("num_key_value_heads")
        or num_attention_heads           # MHA fallback (no GQA)
    ) if not is_ssm else None

    head_dim: int | None = None
    if num_attention_heads and hidden_size:
        head_dim = hidden_size // num_attention_heads

    # ── MoE ───────────────────────────────────────────────────────────────────
    num_experts: int | None = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("moe_num_experts")
    ) if is_moe else None

    num_experts_per_token: int | None = (
        config.get("num_experts_per_tok")
        or config.get("num_selected_experts")
        or config.get("top_k")
    ) if is_moe else None

    # ── SSM geometry ──────────────────────────────────────────────────────────
    state_size: int | None = (
        config.get("state_size")         # Mamba
        or config.get("d_state")
        or config.get("ssm_state_size")
    ) if is_ssm or family == ArchitectureFamily.HYBRID else None

    expand: int | None = config.get("expand") or config.get("expand_factor")
    ssm_expansion_factor: float | None = float(expand) if expand else None

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = config.get("vocab_size", 32000)

    # ── Parameter estimate ────────────────────────────────────────────────────
    param_count = _estimate_params(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        is_ssm=is_ssm,
    )
    model_size_gb = (param_count * 2) / (1024**3)  # bfloat16 = 2 bytes/param

    # ── Extra fields passthrough ──────────────────────────────────────────────
    known_keys = {
        "model_type", "architectures", "num_hidden_layers", "n_layer", "n_layers",
        "hidden_size", "d_model", "n_embd", "intermediate_size", "d_ff",
        "num_attention_heads", "n_head", "num_key_value_heads",
        "num_local_experts", "num_experts", "num_experts_per_tok",
        "state_size", "d_state", "expand", "vocab_size",
    }
    extra = {k: v for k, v in config.items() if k not in known_keys}

    return ArchitectureInfo(
        model_type=model_type,
        architectures=architectures,
        family=family,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        is_moe=is_moe,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        state_size=state_size,
        ssm_expansion_factor=ssm_expansion_factor,
        vocab_size=vocab_size,
        param_count_estimate=param_count,
        model_size_gb_bf16=round(model_size_gb, 2),
        extra=extra,
    )
