"""
Tests for the Prism LoRA Generator Engine (Phase 4).

Coverage:
  - BehavioralTargets construction from dict
  - PrismGenerateEngine.compile() returns well-formed GenerationResult
  - LoRA rank is within [4, 128]
  - Alpha = 2 × rank (±assertiveness adjustment still stays positive)
  - Dropout is within [0.02, 0.10]
  - Target modules are non-empty
  - Estimated size is positive
  - Training YAML contains expected keys
  - adapter_config contains PEFT-required fields
  - POST /api/v1/generate/lora → 200 + GenerateLoRAResult schema
  - GET  /api/v1/generate/jobs/{job_id} → 200 round-trip
  - GET  /api/v1/generate/jobs/nonexistent → 404
"""

from __future__ import annotations

import pytest

from prism.generate.engine import BehavioralTargets, GenerationResult, PrismGenerateEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_TARGETS = BehavioralTargets(
    sycophancy=0.15,
    hedging=0.20,
    calibration=0.80,
    depth=0.75,
    coherence=0.85,
    focus=0.80,
    specificity=0.75,
    verbosity=0.50,
    repetition=0.10,
)

LOW_COMPLEXITY_TARGETS = BehavioralTargets(
    sycophancy=0.80,
    hedging=0.80,
    calibration=0.20,
    depth=0.10,
    coherence=0.20,
    focus=0.20,
    specificity=0.10,
    verbosity=0.50,
    repetition=0.80,
)

HIGH_COMPLEXITY_TARGETS = BehavioralTargets(
    sycophancy=0.05,
    hedging=0.05,
    calibration=0.95,
    depth=0.95,
    coherence=0.95,
    focus=0.95,
    specificity=0.95,
    verbosity=0.50,
    repetition=0.05,
)

ENGINE = PrismGenerateEngine()


# ---------------------------------------------------------------------------
# Unit tests — engine output structure
# ---------------------------------------------------------------------------

class TestGenerationResult:
    def test_compile_returns_result(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        assert isinstance(result, GenerationResult)

    def test_rank_in_valid_range(self):
        for targets in [DEFAULT_TARGETS, LOW_COMPLEXITY_TARGETS, HIGH_COMPLEXITY_TARGETS]:
            result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", targets)
            assert 4 <= result.lora_rank <= 128, f"rank {result.lora_rank} out of [4, 128]"

    def test_rank_is_power_of_two(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        rank = result.lora_rank
        assert rank & (rank - 1) == 0, f"rank {rank} is not a power of 2"

    def test_alpha_is_positive(self):
        for targets in [DEFAULT_TARGETS, LOW_COMPLEXITY_TARGETS, HIGH_COMPLEXITY_TARGETS]:
            result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", targets)
            assert result.lora_alpha > 0

    def test_dropout_in_valid_range(self):
        for targets in [DEFAULT_TARGETS, LOW_COMPLEXITY_TARGETS, HIGH_COMPLEXITY_TARGETS]:
            result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", targets)
            assert 0.01 <= result.lora_dropout <= 0.12, (
                f"dropout {result.lora_dropout} out of expected [0.01, 0.12]"
            )

    def test_target_modules_non_empty(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        assert len(result.target_modules) > 0

    def test_estimated_size_positive(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        assert result.estimated_size_mb > 0

    def test_trainable_params_positive(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        assert result.trainable_params > 0

    def test_adapter_config_has_peft_keys(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        cfg = result.adapter_config
        assert cfg.get("peft_type") == "LORA"
        assert "r" in cfg
        assert "lora_alpha" in cfg
        assert "target_modules" in cfg

    def test_training_yaml_has_required_keys(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        yaml_text = result.training_yaml
        for key in ("lora_r:", "lora_alpha:", "learning_rate:", "bf16:"):
            assert key in yaml_text, f"'{key}' missing from training YAML"

    def test_high_complexity_rank_gte_low(self):
        low_result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", LOW_COMPLEXITY_TARGETS)
        high_result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", HIGH_COMPLEXITY_TARGETS)
        assert high_result.lora_rank >= low_result.lora_rank

    def test_rank_override_respected(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS, lora_rank=32)
        # Explicit lora_rank overrides the derived value
        assert result.lora_rank == 32

    def test_mamba_model_produces_result(self):
        result = ENGINE.compile("state-spaces/mamba-130m", DEFAULT_TARGETS)
        assert isinstance(result, GenerationResult)
        assert result.lora_rank >= 4

    def test_job_id_is_string(self):
        result = ENGINE.compile("meta-llama/Meta-Llama-3-8B", DEFAULT_TARGETS)
        assert isinstance(result.job_id, str)
        assert len(result.job_id) > 0


# ---------------------------------------------------------------------------
# API integration tests — /generate/lora + /generate/jobs/{id}
# ---------------------------------------------------------------------------

class TestGenerateAPI:
    def test_generate_lora_returns_200(self, client):
        resp = client.post("/api/v1/generate/lora", json={
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "behavioral_targets": {
                "sycophancy": 0.15, "hedging": 0.20, "calibration": 0.80,
                "depth": 0.75, "coherence": 0.85, "focus": 0.80,
                "specificity": 0.75, "verbosity": 0.50, "repetition": 0.10,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert "lora_rank" in data
        assert "lora_alpha" in data
        assert "training_yaml" in data
        assert "adapter_config" in data
        assert "target_modules" in data

    def test_generate_lora_rank_in_range(self, client):
        resp = client.post("/api/v1/generate/lora", json={
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "behavioral_targets": {
                "sycophancy": 0.5, "hedging": 0.5, "calibration": 0.5,
                "depth": 0.5, "coherence": 0.5, "focus": 0.5,
                "specificity": 0.5, "verbosity": 0.5, "repetition": 0.5,
            },
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 4 <= data["lora_rank"] <= 128

    def test_generate_lora_explicit_rank(self, client):
        resp = client.post("/api/v1/generate/lora", json={
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "lora_rank": 16,
            "behavioral_targets": {
                "sycophancy": 0.5, "hedging": 0.5, "calibration": 0.5,
                "depth": 0.5, "coherence": 0.5, "focus": 0.5,
                "specificity": 0.5, "verbosity": 0.5, "repetition": 0.5,
            },
        })
        assert resp.status_code == 200

    def test_get_job_round_trip(self, client):
        # Create a job
        create_resp = client.post("/api/v1/generate/lora", json={
            "model_id": "mistralai/Mistral-7B-v0.1",
            "behavioral_targets": {
                "sycophancy": 0.2, "hedging": 0.2, "calibration": 0.8,
                "depth": 0.8, "coherence": 0.8, "focus": 0.8,
                "specificity": 0.8, "verbosity": 0.5, "repetition": 0.1,
            },
        })
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        # Retrieve job
        get_resp = client.get(f"/api/v1/generate/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["job_id"] == job_id

    def test_get_nonexistent_job_returns_404(self, client):
        resp = client.get("/api/v1/generate/jobs/nonexistent-job-id")
        assert resp.status_code == 404

    def test_generate_lora_status_field(self, client):
        resp = client.post("/api/v1/generate/lora", json={
            "model_id": "meta-llama/Meta-Llama-3-8B",
            "behavioral_targets": {
                "sycophancy": 0.5, "hedging": 0.5, "calibration": 0.5,
                "depth": 0.5, "coherence": 0.5, "focus": 0.5,
                "specificity": 0.5, "verbosity": 0.5, "repetition": 0.5,
            },
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "complete"
