"""
LoRA target module mappings by model architecture type.

Each entry maps a HuggingFace `model_type` string (from config.json) to the
list of sub-module names that should be wrapped with LoRA adapters.

Guidelines for target selection:
- For full-rank fine-tuning fidelity: include all projection matrices.
- For lightweight/fast adapters: use only Q + V projections.
- For SSMs: target the state-space projection matrices instead.

The Prism defaults to the "full" set for maximum behavioral control.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Primary mapping: model_type → LoRA target modules
# ---------------------------------------------------------------------------
LORA_TARGET_MAP: dict[str, list[str]] = {
    # ── LLaMA family ────────────────────────────────────────────────────────
    # LLaMA 1/2/3, Code-LLaMA, Vicuna, Alpaca, WizardLM, …
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── Mistral / Mixtral ────────────────────────────────────────────────────
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── Qwen2 / Qwen2-MoE ───────────────────────────────────────────────────
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2_moe": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2_vl": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── Phi family ──────────────────────────────────────────────────────────
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "phi3small": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    # ── Gemma family ────────────────────────────────────────────────────────
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── DeepSeek family ─────────────────────────────────────────────────────
    "deepseek_v2": [
        "q_proj", "kv_a_proj_with_mqa", "q_a_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "deepseek_v3": [
        "q_proj", "kv_a_proj_with_mqa", "q_a_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # ── Falcon (attention-based) ─────────────────────────────────────────────
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "RefinedWebModel": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # ── Mamba / SSM family ───────────────────────────────────────────────────
    # Pure SSM — no attention heads; target state-space projection matrices.
    "mamba": ["in_proj", "out_proj", "x_proj", "dt_proj"],
    "mamba2": ["in_proj", "out_proj", "z_proj", "dt_proj"],
    # ── Falcon-Mamba (SSM variant of Falcon) ─────────────────────────────────
    "falcon_mamba": ["in_proj", "out_proj", "x_proj", "dt_proj"],
    # ── Jamba (Mamba + Attention hybrid) ────────────────────────────────────
    "jamba": [
        # Attention layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        # Mamba layers
        "in_proj", "out_proj",
    ],
    # ── Zamba ────────────────────────────────────────────────────────────────
    "zamba": ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj"],
    # ── GPT-2 family ─────────────────────────────────────────────────────────
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    # ── GPT-NeoX / Pythia ────────────────────────────────────────────────────
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # ── GPT-J ────────────────────────────────────────────────────────────────
    "gptj": ["q_proj", "v_proj", "fc_in", "fc_out"],
    # ── OPT ──────────────────────────────────────────────────────────────────
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    # ── BLOOM ────────────────────────────────────────────────────────────────
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # ── StarCoder2 ────────────────────────────────────────────────────────────
    "starcoder2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── T5 family (encoder-decoder) ──────────────────────────────────────────
    "t5": ["q", "v", "k", "o", "wi", "wo"],
    "mt5": ["q", "v", "k", "o", "wi", "wo"],
    "flan-t5": ["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
    # ── BERT family ───────────────────────────────────────────────────────────
    "bert": ["query", "value", "key", "dense"],
    "roberta": ["query", "value", "key", "dense"],
    "deberta-v2": ["query_proj", "key_proj", "value_proj", "pos_proj"],
    # ── InternLM ─────────────────────────────────────────────────────────────
    "internlm2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── Yi ────────────────────────────────────────────────────────────────────
    "yi": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── Baichuan ─────────────────────────────────────────────────────────────
    "baichuan": ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # ── ChatGLM ───────────────────────────────────────────────────────────────
    "chatglm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}

# ---------------------------------------------------------------------------
# Minimal (Q+V only) targets — used when `lora_rank` is very small or for
# quick diagnostic adapters where full coverage is unnecessary.
# ---------------------------------------------------------------------------
MINIMAL_LORA_TARGETS: dict[str, list[str]] = {
    "llama": ["q_proj", "v_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "qwen2": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj"],
    "gemma": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "gpt_neox": ["query_key_value"],
    "falcon": ["query_key_value"],
    "mamba": ["in_proj", "out_proj"],
    "falcon_mamba": ["in_proj", "out_proj"],
    "jamba": ["q_proj", "v_proj", "in_proj"],
    "t5": ["q", "v"],
    "bert": ["query", "value"],
}

# ---------------------------------------------------------------------------
# Fallback defaults when model_type is unknown
# ---------------------------------------------------------------------------
_ATTENTION_FALLBACK = ["q_proj", "v_proj", "k_proj", "o_proj"]
_SSM_FALLBACK = ["in_proj", "out_proj"]


def get_lora_targets(
    model_type: str,
    *,
    minimal: bool = False,
    fallback_is_attention: bool = True,
) -> list[str]:
    """
    Return the recommended LoRA target modules for the given HF model_type.

    Args:
        model_type: The `model_type` string from HuggingFace config.json.
        minimal: If True, return Q+V only targets (smaller adapter, faster training).
        fallback_is_attention: Used only when model_type is unknown — if True,
            fall back to standard attention projections; otherwise use SSM targets.

    Returns:
        Ordered list of sub-module name strings suitable for PEFT ``target_modules``.
    """
    source = MINIMAL_LORA_TARGETS if minimal else LORA_TARGET_MAP
    key = model_type.lower()

    # Direct hit
    if key in source:
        return list(source[key])

    # Prefix / substring match (e.g. "llama3" → "llama")
    for canonical_key in source:
        if key.startswith(canonical_key) or canonical_key in key:
            return list(source[canonical_key])

    # Unknown architecture — use safe fallback
    return list(_ATTENTION_FALLBACK if fallback_is_attention else _SSM_FALLBACK)
