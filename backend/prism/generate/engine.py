"""
Prism Generate Engine — Phase 4.

Accepts 9 behavioral slider values (0.0–1.0 per dimension) + architecture info,
maps them to LoRA hyperparameters, and emits a PEFT-compatible adapter config
plus a ready-to-use training YAML.

The mock engine is deterministic: same model_id + same targets → same output.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BehavioralTargets:
    sycophancy: float = 0.5
    hedging: float = 0.5
    calibration: float = 0.5
    depth: float = 0.5
    coherence: float = 0.5
    focus: float = 0.5
    specificity: float = 0.5
    verbosity: float = 0.5
    repetition: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {
            "sycophancy": self.sycophancy,
            "hedging": self.hedging,
            "calibration": self.calibration,
            "depth": self.depth,
            "coherence": self.coherence,
            "focus": self.focus,
            "specificity": self.specificity,
            "verbosity": self.verbosity,
            "repetition": self.repetition,
        }


@dataclass
class GenerationResult:
    job_id: str
    model_id: str
    status: str                          # "complete" | "failed"
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: list[str]
    adapter_config: dict[str, Any]       # PEFT adapter_config.json contents
    training_yaml: str                   # ready-to-paste training config
    trainable_params: int
    estimated_size_mb: float


# ---------------------------------------------------------------------------
# Rank / alpha / dropout derivation from behavioral targets
# ---------------------------------------------------------------------------

_DEFAULT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

_MINIMAL_TARGETS = ["q_proj", "v_proj"]


def _derive_rank(targets: BehavioralTargets, base_rank: int) -> int:
    """
    Higher depth / specificity / coherence → more expressive adapter → higher rank.
    Rank is snapped to the nearest power of 2 in [4, 128].
    """
    complexity = (
        targets.depth * 0.35
        + targets.specificity * 0.25
        + targets.coherence * 0.20
        + targets.calibration * 0.20
    )
    # Map 0–1 complexity to rank range 4–128 on log scale
    import math
    log_rank = math.log2(4) + complexity * (math.log2(128) - math.log2(4))
    raw_rank = 2 ** round(log_rank)
    # Clamp
    return max(4, min(128, raw_rank))


def _derive_alpha(rank: int, targets: BehavioralTargets) -> float:
    """
    Standard convention: alpha = 2 × rank.
    Adjust upward when targeting lower sycophancy (more assertive steering).
    """
    base = 2.0 * rank
    assertiveness_boost = (1.0 - targets.sycophancy) * 0.5  # 0–0.5 multiplier
    return round(base * (1.0 + assertiveness_boost), 1)


def _derive_dropout(targets: BehavioralTargets) -> float:
    """Higher coherence / focus targets → less dropout (preserve fine structure)."""
    stability = (targets.coherence + targets.focus) / 2.0
    return round(0.10 - stability * 0.08, 3)   # 0.02–0.10


def _derive_target_modules(targets: BehavioralTargets) -> list[str]:
    """Minimal targets suffice for low-complexity tweaks; full list otherwise."""
    complexity = targets.depth * 0.4 + targets.specificity * 0.3 + targets.coherence * 0.3
    return _DEFAULT_TARGETS if complexity >= 0.45 else _MINIMAL_TARGETS


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------

class PrismGenerateEngine:
    """
    Phase 4 LoRA generation engine.

    Derives LoRA hyperparameters from the 9 behavioral sliders via the
    fiber-space projection heuristics, then emits a PEFT-compatible
    adapter_config.json + a ready-to-use training YAML.
    """

    def compile(
        self,
        model_id: str,
        targets: BehavioralTargets,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        output_path: Path | None = None,
    ) -> GenerationResult:
        # --- Derive hyperparameters ------------------------------------------
        rank    = lora_rank  or _derive_rank(targets, base_rank=16)
        alpha   = lora_alpha or _derive_alpha(rank, targets)
        dropout = _derive_dropout(targets)
        modules = _derive_target_modules(targets)

        # --- PEFT adapter_config.json ----------------------------------------
        adapter_config: dict[str, Any] = {
            "base_model_name_or_path": model_id,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": False,
            "init_lora_weights": True,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": rank,
            "revision": "main",
            "target_modules": modules,
            "task_type": "CAUSAL_LM",
            # Behavioral provenance metadata
            "prism_behavioral_targets": targets.as_dict(),
            "prism_version": "1.0.0",
        }

        # --- Trainable parameter estimate ------------------------------------
        # Rough heuristic: each target module contributes 2 × (hidden × rank) params
        # We don't know hidden_size here, so estimate from model name heuristics
        hidden_size = _estimate_hidden_size(model_id)
        trainable = len(modules) * 2 * hidden_size * rank
        size_mb = round(trainable * 4 / (1024 ** 2), 2)   # bfloat16 ≈ 4 bytes/param approx

        # --- Training YAML ---------------------------------------------------
        yaml = _build_training_yaml(
            model_id=model_id,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            modules=modules,
            targets=targets,
        )

        job_id = str(uuid.uuid4())[:8]

        return GenerationResult(
            job_id=job_id,
            model_id=model_id,
            status="complete",
            lora_rank=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=modules,
            adapter_config=adapter_config,
            training_yaml=yaml,
            trainable_params=trainable,
            estimated_size_mb=size_mb,
        )


def _estimate_hidden_size(model_id: str) -> int:
    """Rough hidden size guess from model name token counts."""
    lower = model_id.lower()
    if any(x in lower for x in ["70b", "72b", "65b"]):
        return 8192
    if any(x in lower for x in ["34b", "32b", "30b"]):
        return 6144
    if any(x in lower for x in ["13b", "14b"]):
        return 5120
    if any(x in lower for x in ["7b", "8b"]):
        return 4096
    if any(x in lower for x in ["3b", "2b"]):
        return 2560
    if any(x in lower for x in ["1b", "130m", "370m"]):
        return 1024
    return 4096   # sensible default


def _build_training_yaml(
    model_id: str,
    rank: int,
    alpha: float,
    dropout: float,
    modules: list[str],
    targets: BehavioralTargets,
) -> str:
    """Build a LLaMA-Factory / Axolotl-compatible training config YAML."""
    mod_list = "\n".join(f"  - {m}" for m in modules)
    target_lines = "\n".join(
        f"  {k}: {round(v, 3)}" for k, v in targets.as_dict().items()
    )
    # Determine reasonable batch size and gradient accum steps
    grad_accum = 4 if rank >= 32 else 8
    return f"""\
# ============================================================
# Prism-generated LoRA training config
# Model:  {model_id}
# Phase:  4 — Behavioral Precision Tuning
# ============================================================

model_name_or_path: {model_id}
trust_remote_code: true

# LoRA hyperparameters (derived from behavioral targets)
lora_r: {rank}
lora_alpha: {alpha}
lora_dropout: {dropout}
lora_target_modules:
{mod_list}
lora_bias: none

# Task
task_type: CAUSAL_LM
peft_type: LORA

# Training
output_dir: ./prism-lora-{model_id.replace("/", "--")}
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: {grad_accum}
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
fp16: false
bf16: true
gradient_checkpointing: true
dataloader_num_workers: 4

# Logging
logging_steps: 10
save_strategy: epoch
evaluation_strategy: epoch
report_to: tensorboard

# Dataset — replace with your data path
dataset: ./data/train.jsonl
dataset_format: alpaca

# ---- Prism behavioral targets (for provenance) ----
# prism_targets:
{chr(10).join("# " + line for line in target_lines.splitlines())}
"""
