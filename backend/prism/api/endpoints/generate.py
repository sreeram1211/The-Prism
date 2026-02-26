"""
Prism Generate API endpoints — Phase 4.

POST /api/v1/generate/lora         — compile a LoRA adapter config from behavioral targets
GET  /api/v1/generate/jobs/{id}    — poll job status (always "complete" in mock)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from prism.generate.engine import BehavioralTargets, PrismGenerateEngine
from prism.schemas.models import GenerateRequest, GenerateLoRAResult

router = APIRouter(prefix="/generate", tags=["Prism Generate"])

_engine = PrismGenerateEngine()

# In-memory job store (Phase 4 mock — no persistence)
_jobs: dict[str, GenerateLoRAResult] = {}


@router.post(
    "/lora",
    response_model=GenerateLoRAResult,
    summary="Compile a precision LoRA adapter config from behavioral targets",
    description=(
        "Accepts 9 behavioral slider values (0.0–1.0) and derives optimal LoRA "
        "hyperparameters via Prism's fiber-space projection heuristics. Returns a "
        "PEFT-compatible adapter_config.json and a ready-to-use training YAML."
    ),
    status_code=status.HTTP_200_OK,
)
def generate_lora(body: GenerateRequest) -> GenerateLoRAResult:
    targets = BehavioralTargets(
        sycophancy=body.behavioral_targets.sycophancy,
        hedging=body.behavioral_targets.hedging,
        calibration=body.behavioral_targets.calibration,
        depth=body.behavioral_targets.depth,
        coherence=body.behavioral_targets.coherence,
        focus=body.behavioral_targets.focus,
        specificity=body.behavioral_targets.specificity,
        verbosity=body.behavioral_targets.verbosity,
        repetition=body.behavioral_targets.repetition,
    )

    result = _engine.compile(
        model_id=body.model_id,
        targets=targets,
        lora_rank=body.lora_rank if body.lora_rank != 16 else None,
        lora_alpha=body.lora_alpha if body.lora_alpha != 32.0 else None,
    )

    out = GenerateLoRAResult(
        job_id=result.job_id,
        status=result.status,
        model_id=result.model_id,
        lora_rank=result.lora_rank,
        lora_alpha=result.lora_alpha,
        lora_dropout=result.lora_dropout,
        target_modules=result.target_modules,
        adapter_config=result.adapter_config,
        training_yaml=result.training_yaml,
        trainable_params=result.trainable_params,
        estimated_size_mb=result.estimated_size_mb,
    )
    _jobs[result.job_id] = out
    return out


@router.get(
    "/jobs/{job_id}",
    response_model=GenerateLoRAResult,
    summary="Poll a LoRA generation job",
)
def get_job_status(job_id: str) -> GenerateLoRAResult:
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "job_not_found", "job_id": job_id},
        )
    return _jobs[job_id]
