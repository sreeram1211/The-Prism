"""
Pydantic v2 request/response models for all Prism API endpoints.

Stubs for Phase 2–5 features are included here so the routing layer is
fully typed from day one; they will be fleshed out in subsequent phases.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# Common
# ===========================================================================

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    phase: str = "1 — Auto-Resolver"


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    code: str | None = None


# ===========================================================================
# Resolver (Phase 1)
# ===========================================================================

class ResolveRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="HuggingFace model repository ID, e.g. 'meta-llama/Meta-Llama-3-8B'.",
        examples=["mistralai/Mistral-7B-Instruct-v0.2"],
        min_length=3,
    )
    revision: str = Field(
        default="main",
        description="Git revision / branch / tag on HuggingFace Hub.",
    )
    hf_token: str | None = Field(
        default=None,
        description="Optional HuggingFace access token for gated models.",
    )

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model_id must not be empty")
        # Basic sanity check — HF IDs are "namespace/name" or just "name"
        if "/" in v:
            parts = v.split("/")
            if len(parts) != 2 or not all(parts):
                raise ValueError("model_id must be in 'namespace/model-name' format")
        return v


class ModelResolverResponse(BaseModel):
    """Full architecture descriptor returned by the Auto-Resolver."""

    model_id: str
    revision: str

    # Core architecture
    model_type: str
    architectures: list[str]
    family: str = Field(description="Architecture family: attention | ssm | hybrid | encoder_decoder")

    # Layer geometry
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int

    # Attention geometry (null for SSMs)
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    uses_gqa: bool

    # MoE
    is_moe: bool
    num_experts: int | None
    num_experts_per_token: int | None

    # SSM
    state_size: int | None
    ssm_expansion_factor: float | None

    # Vocab
    vocab_size: int

    # Size estimates
    param_count_estimate: int = Field(description="Approximate total parameter count")
    model_size_gb_bf16: float = Field(description="Estimated model size in GB at bfloat16")

    # LoRA recommendations
    lora_rank_recommendation: int
    lora_targets: list[str] = Field(description="Full LoRA target module list")
    lora_targets_minimal: list[str] = Field(description="Minimal Q+V only targets")

    model_config = {"json_schema_extra": {
        "example": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "revision": "main",
            "model_type": "mistral",
            "architectures": ["MistralForCausalLM"],
            "family": "attention",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "uses_gqa": True,
            "is_moe": False,
            "num_experts": None,
            "num_experts_per_token": None,
            "state_size": None,
            "ssm_expansion_factor": None,
            "vocab_size": 32000,
            "param_count_estimate": 7241732096,
            "model_size_gb_bf16": 13.49,
            "lora_rank_recommendation": 16,
            "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_targets_minimal": ["q_proj", "v_proj"],
        }
    }}


# ===========================================================================
# Scan (Phase 2 — CLI + Web UI)
# ===========================================================================

class ScanDimension(str, Enum):
    SYCOPHANCY = "sycophancy"
    HEDGING = "hedging"
    CALIBRATION = "calibration"
    DEPTH = "depth"
    COHERENCE = "coherence"
    FOCUS = "focus"
    SPECIFICITY = "specificity"
    VERBOSITY = "verbosity"
    REPETITION = "repetition"


class ScanRequest(BaseModel):
    model_id: str = Field(..., description="Model to scan")
    prompts: list[str] | None = Field(
        default=None,
        description="Custom evaluation prompts. Uses Prism defaults if omitted.",
    )
    dimensions: list[ScanDimension] | None = Field(
        default=None,
        description="Dimensions to evaluate. Evaluates all 9 if omitted.",
    )


class DimensionScore(BaseModel):
    dimension: ScanDimension
    score: float = Field(ge=0.0, le=1.0)
    raw_value: float
    interpretation: str


class ScanResult(BaseModel):
    model_id: str
    scan_id: str | None = Field(default=None, description="Persisted scan ID (Phase 6+)")
    scores: list[DimensionScore]
    geometric_separation_ratio: float = Field(
        description="16-dim fiber space separation ratio (125x–1376x range)"
    )
    scan_duration_ms: float


# ===========================================================================
# Generate / LoRA (Phase 3)
# ===========================================================================

class BehavioralTarget(BaseModel):
    """Single slider value for the Generate interface (0.0–1.0)."""

    sycophancy: float = Field(default=0.5, ge=0.0, le=1.0)
    hedging: float = Field(default=0.5, ge=0.0, le=1.0)
    calibration: float = Field(default=0.5, ge=0.0, le=1.0)
    depth: float = Field(default=0.5, ge=0.0, le=1.0)
    coherence: float = Field(default=0.5, ge=0.0, le=1.0)
    focus: float = Field(default=0.5, ge=0.0, le=1.0)
    specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    verbosity: float = Field(default=0.5, ge=0.0, le=1.0)
    repetition: float = Field(default=0.5, ge=0.0, le=1.0)


class GenerateRequest(BaseModel):
    model_id: str
    revision: str = "main"
    behavioral_targets: BehavioralTarget
    lora_rank: int | None = Field(
        default=None,
        ge=4,
        le=256,
        description="LoRA rank. If omitted, derived automatically from behavioral targets.",
    )
    lora_alpha: float | None = Field(
        default=None,
        gt=0.0,
        description="LoRA alpha. If omitted, derived automatically as 2 × rank.",
    )
    output_path: str | None = None


class GenerateResult(BaseModel):
    job_id: str
    status: str
    model_id: str
    adapter_path: str | None
    estimated_duration_s: float | None


class GenerateLoRAResult(BaseModel):
    """Phase 4: full LoRA config result returned by /generate/lora."""

    job_id: str
    status: str
    model_id: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: list[str]
    adapter_config: dict[str, Any]
    training_yaml: str
    trainable_params: int
    estimated_size_mb: float


# ===========================================================================
# Monitor / Telemetry (Phase 5)
# ===========================================================================

class MonitorSessionCreate(BaseModel):
    model_id: str
    layers: list[int] | None = Field(
        default=None,
        description="Layer indices to monitor. Defaults to [0, 8, 16, 24].",
    )
    prompt: str = Field(
        default="Tell me about large language models.",
        description="Prompt to generate activations from.",
    )


class MonitorSession(BaseModel):
    session_id: str
    model_id: str
    websocket_url: str
    telemetry_layers: list[int] = Field(
        description="Layer indices being monitored by the Proprioceptive Nervous System"
    )


# ===========================================================================
# Agent (Phase 5)
# ===========================================================================

class AgentChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    use_memory: bool = True
    memory_top_k: int = Field(default=5, ge=1, le=20)


class AgentChatResponse(BaseModel):
    session_id: str
    reply: str
    memory_hits: int
    alpha_prime: float | None = Field(
        default=None,
        description="RSI engine acceleration metric (α') — improvement rate across sessions",
    )


# ===========================================================================
# Persistence / History (Phase 6)
# ===========================================================================

class ScanHistoryItem(BaseModel):
    scan_id: str
    model_id: str
    created_at: str = Field(description="ISO-8601 UTC timestamp")
    duration_ms: float
    geometric_separation_ratio: float
    top_score: DimensionScore


class ScanHistoryResponse(BaseModel):
    items: list[ScanHistoryItem]
    total: int
    limit: int
    offset: int


# ===========================================================================
# Comparison (Phase 6)
# ===========================================================================

class CompareRequest(BaseModel):
    scan_a: str = Field(..., description="Scan ID of the first model")
    scan_b: str = Field(..., description="Scan ID of the second model")


class DimensionDelta(BaseModel):
    dimension: str
    score_a: float
    score_b: float
    delta: float = Field(description="score_b − score_a")
    direction: str = Field(description="'improved' | 'regressed' | 'neutral'")


class CompareResult(BaseModel):
    scan_id_a: str
    scan_id_b: str
    model_a: str
    model_b: str
    deltas: list[DimensionDelta]
    composite_distance: float = Field(description="L2 distance in 9-dim score space")
    winner: str = Field(description="'a' | 'b' | 'tie'")


# ===========================================================================
# Dashboard (Phase 6)
# ===========================================================================

class DashboardStats(BaseModel):
    total_scans: int
    total_jobs: int
    total_sessions: int
    unique_models: int
    recent_scans: list[ScanHistoryItem]
