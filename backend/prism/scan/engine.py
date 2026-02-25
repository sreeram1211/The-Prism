"""
Prism Scan Engine — Phase 2 stub.

The full engine interfaces with the proprietary C++/Python behavioral probe
library to compute ROC-AUC-verified scores across the 9 behavioral dimensions.

The 16-dimensional geometric separation ratios (125x–1,376x) are derived
from the fiber space projection and returned alongside per-dimension scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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

    @classmethod
    def all(cls) -> list["ScanDimension"]:
        return list(cls)


@dataclass
class DimensionScore:
    dimension: ScanDimension
    score: float          # 0.0–1.0 normalized
    raw_value: float      # unnormalized probe output
    interpretation: str   # human-readable verdict


@dataclass
class ScanReport:
    model_id: str
    scores: list[DimensionScore]
    geometric_separation_ratio: float  # 16-dim fiber space
    scan_duration_ms: float


class PrismScanEngine:
    """
    Phase 2 placeholder.

    In Phase 2 this class will:
      1. Load a set of curated evaluation prompts (or accept custom ones).
      2. Run inference through the target model.
      3. Project activations into the 16-dimensional fiber space.
      4. Compute ROC-AUC verified probe scores per dimension.
      5. Return a ScanReport.
    """

    def scan(self, model_id: str, prompts: list[str] | None = None) -> ScanReport:
        raise NotImplementedError("Prism Scan is implemented in Phase 2.")
