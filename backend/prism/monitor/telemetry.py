"""
Prism Monitor Telemetry — Phase 5.

Streams simulated ActivationFrame objects via an async generator, mimicking
the sub-millisecond per-token activation capture from the 4-layer
Proprioceptive Nervous System (PNS).

The mock is seeded from the model ID for reproducibility in demos.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator


@dataclass
class ActivationFrame:
    """One telemetry frame — emitted once per simulated generated token."""

    token_index: int
    token_text: str
    timestamp_us: int                          # µs since session start
    layer_activations: dict[int, list[float]]  # layer_idx → 8-float activation sample
    reflex_override: bool                      # True when steering overrides reflex
    manifold_coords: list[float]               # 16-dim fiber-space position
    dimension_drift: dict[str, float]          # per-dimension drift from baseline

    def as_dict(self) -> dict:
        return {
            "token_index": self.token_index,
            "token_text": self.token_text,
            "timestamp_us": self.timestamp_us,
            "layer_activations": self.layer_activations,
            "reflex_override": self.reflex_override,
            "manifold_coords": self.manifold_coords,
            "dimension_drift": self.dimension_drift,
        }


@dataclass
class MonitorSession:
    session_id: str
    model_id: str
    prompt: str
    monitored_layers: list[int] = field(default_factory=lambda: [0, 8, 16, 24])
    frame_count: int = 0
    total_reflex_overrides: int = 0
    start_time_us: int = field(default_factory=lambda: int(time.time() * 1_000_000))


# ---------------------------------------------------------------------------
# Token vocabulary for the mock stream
# ---------------------------------------------------------------------------

_TOKENS = [
    "The", "model", "learns", "to", "generate", "coherent", "text",
    "by", "predicting", "the", "next", "token", "in", "a", "sequence",
    "using", "attention", "mechanisms", "and", "feedforward", "layers",
    ".", "Transformers", "encode", "semantic", "structure", "through",
    "high-dimensional", "representations", ".", "Fine-tuning", "with",
    "LoRA", "adapters", "introduces", "low-rank", "weight", "deltas",
    "that", "steer", "behaviour", "efficiently", ".", "The", "Prism",
    "manifold", "captures", "these", "dynamics", "in", "a", "16-dim",
    "fiber", "space", ".",
]

_DIMS = [
    "sycophancy", "hedging", "calibration", "depth",
    "coherence", "focus", "specificity", "verbosity", "repetition",
]


class PrismTelemetryStream:
    """
    Phase 5 mock activation telemetry stream.

    Yields deterministically-seeded ActivationFrame objects at a controlled
    rate to simulate sub-millisecond per-token activation capture from the
    4-layer Proprioceptive Nervous System.
    """

    async def stream(
        self,
        session: MonitorSession,
        max_tokens: int = 60,
        tokens_per_second: float = 8.0,
    ) -> AsyncGenerator[ActivationFrame, None]:
        seed = int(hashlib.md5(session.model_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        delay = 1.0 / tokens_per_second
        layers = session.monitored_layers or [0, 8, 16, 24]

        # Baseline manifold coords — fixed per model
        baseline_coords = [rng.gauss(0.0, 1.0) for _ in range(16)]

        # Baseline dimension values
        baseline_dims = {d: rng.uniform(0.3, 0.7) for d in _DIMS}

        for i in range(max_tokens):
            await asyncio.sleep(delay)

            token_text = rng.choice(_TOKENS)
            timestamp_us = session.start_time_us + int(i / tokens_per_second * 1_000_000)

            # Layer activations: sinusoidal + noise, different per layer
            layer_activations: dict[int, list[float]] = {}
            for layer_idx in layers:
                phase = layer_idx * 0.3 + i * 0.15
                acts = [
                    round(math.sin(phase + j * 0.8) * 0.5 + rng.gauss(0.0, 0.05), 4)
                    for j in range(8)
                ]
                layer_activations[layer_idx] = acts

            # Reflex override: ~8% probability, slightly higher mid-sequence
            override_prob = 0.08 + 0.04 * math.sin(i / max_tokens * math.pi)
            reflex_override = rng.random() < override_prob
            if reflex_override:
                session.total_reflex_overrides += 1

            # Manifold coords: slow drift from baseline
            drift_scale = 0.02
            manifold_coords = [
                round(baseline_coords[k] + rng.gauss(0.0, drift_scale) * math.sqrt(i + 1), 4)
                for k in range(16)
            ]

            # Dimension drift: small per-step random walk
            dimension_drift = {
                d: round(rng.gauss(0.0, 0.01), 4)
                for d in _DIMS
            }

            frame = ActivationFrame(
                token_index=i,
                token_text=token_text,
                timestamp_us=timestamp_us,
                layer_activations=layer_activations,
                reflex_override=reflex_override,
                manifold_coords=manifold_coords,
                dimension_drift=dimension_drift,
            )
            session.frame_count += 1
            yield frame
