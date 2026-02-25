"""Prism Auto-Resolver: HuggingFace model architecture detection and LoRA targeting."""

from .auto_resolver import PrismAutoResolver
from .arch_detector import ArchitectureInfo, ArchitectureFamily, detect_architecture
from .lora_targets import get_lora_targets, LORA_TARGET_MAP

__all__ = [
    "PrismAutoResolver",
    "ArchitectureInfo",
    "ArchitectureFamily",
    "detect_architecture",
    "get_lora_targets",
    "LORA_TARGET_MAP",
]
