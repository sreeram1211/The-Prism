"""
Rich renderers for all Prism CLI output.

All functions return Rich renderables (Panel, Table, …) rather than
printing directly so that callers control layout and can compose them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from prism.cli.display.theme import (
    FAMILY_STYLES,
    make_bar,
    score_style,
    styled_family,
)

if TYPE_CHECKING:
    from prism.resolver.auto_resolver import ModelResolverResult
    from prism.scan.engine import ScanReport


# ---------------------------------------------------------------------------
# Resolve result renderer
# ---------------------------------------------------------------------------

def render_resolver_result(result: "ModelResolverResult") -> Group:
    """Render a ModelResolverResult as a Rich Group of panels."""
    arch = result.arch

    # ── Architecture info table ───────────────────────────────────────────
    arch_table = Table(show_header=False, box=None, padding=(0, 1))
    arch_table.add_column("Key",   style="info.key",   no_wrap=True)
    arch_table.add_column("Value", style="info.value", no_wrap=False)

    arch_table.add_row("Model ID",    f"[bold]{result.model_id}[/]")
    arch_table.add_row("Revision",    result.revision)
    arch_table.add_row("Type",        f"[bold]{arch.model_type}[/] ({', '.join(arch.architectures) or '—'})")
    arch_table.add_row("Family",      styled_family(arch.family.value))
    arch_table.add_row("Layers",      str(arch.num_hidden_layers))
    arch_table.add_row("Hidden size", f"{arch.hidden_size:,}")
    arch_table.add_row("FFN size",    f"{arch.intermediate_size:,}")

    if arch.num_attention_heads is not None:
        kv_str = ""
        if arch.num_key_value_heads and arch.num_key_value_heads != arch.num_attention_heads:
            kv_str = f" / [lora.rank]{arch.num_key_value_heads} KV[/] [dim white](GQA)[/]"
        arch_table.add_row("Attention heads", f"{arch.num_attention_heads}{kv_str}")
        if arch.head_dim:
            arch_table.add_row("Head dim", str(arch.head_dim))
    else:
        arch_table.add_row("Attention heads", "[dim]— (SSM, no attention)[/]")

    if arch.is_moe:
        arch_table.add_row(
            "MoE experts",
            f"[bold yellow]{arch.num_experts}[/] total / "
            f"[bold yellow]{arch.num_experts_per_token}[/] active per token",
        )

    if arch.state_size:
        arch_table.add_row("SSM state size", str(arch.state_size))
    if arch.ssm_expansion_factor:
        arch_table.add_row("SSM expand factor", f"{arch.ssm_expansion_factor}×")

    arch_table.add_row("Vocab size",   f"{arch.vocab_size:,}")
    arch_table.add_row(
        "Parameters",
        f"[bold]~{arch.param_count_estimate / 1e9:.2f}B[/]",
    )
    arch_table.add_row(
        "Size (BF16)",
        f"[bold]{arch.model_size_gb_bf16:.2f} GB[/]",
    )

    arch_panel = Panel(
        arch_table,
        title="[brand]Architecture Profile[/]",
        border_style="bright_cyan",
        padding=(0, 1),
    )

    # ── LoRA targets table ────────────────────────────────────────────────
    lora_table = Table(show_header=False, box=None, padding=(0, 1))
    lora_table.add_column("Label", style="info.key",    no_wrap=True)
    lora_table.add_column("Value", style="info.value",  no_wrap=False)

    full_targets = "  ".join(f"[lora.target]{t}[/]" for t in result.lora_targets)
    min_targets  = "  ".join(f"[lora.target.min]{t}[/]" for t in result.lora_targets_minimal)

    lora_table.add_row(
        f"Rank [lora.rank]{arch.lora_rank_recommendation}[/] (recommended)",
        full_targets,
    )
    lora_table.add_row("Minimal (Q+V)", min_targets)

    lora_panel = Panel(
        lora_table,
        title="[brand]LoRA Targets[/]",
        border_style="cyan",
        padding=(0, 1),
    )

    return Group(arch_panel, lora_panel)


# ---------------------------------------------------------------------------
# Scan result renderer
# ---------------------------------------------------------------------------

# Metadata per dimension: (display_name, higher_is_better, direction_symbol)
_DIM_META: dict[str, tuple[str, bool, str]] = {
    "sycophancy":  ("Sycophancy",  False, "↓"),
    "hedging":     ("Hedging",     False, "↓"),
    "calibration": ("Calibration", True,  "↑"),
    "depth":       ("Depth",       True,  "↑"),
    "coherence":   ("Coherence",   True,  "↑"),
    "focus":       ("Focus",       True,  "↑"),
    "specificity": ("Specificity", True,  "↑"),
    "verbosity":   ("Verbosity",   None,  "◎"),  # None = optimal at 0.5
    "repetition":  ("Repetition",  False, "↓"),
}


def _verbosity_style(score: float) -> str:
    diff = abs(score - 0.5)
    if diff <= 0.15:
        return "score.good"
    if diff <= 0.30:
        return "score.mid"
    return "score.bad"


def render_scan_result(report: "ScanReport") -> Group:
    """Render a ScanReport as a Rich Group of panels."""

    # ── Scores table ─────────────────────────────────────────────────────
    score_table = Table(
        show_header=True,
        header_style="bold white",
        border_style="bright_cyan",
        show_lines=False,
        padding=(0, 1),
    )
    score_table.add_column("Dimension",    style="score.label", width=14, no_wrap=True)
    score_table.add_column("Dir",          justify="center",    width=3,  no_wrap=True)
    score_table.add_column("Score",        justify="right",     width=6,  no_wrap=True)
    score_table.add_column("Profile",      width=24,            no_wrap=True)
    score_table.add_column("Interpretation", style="dim white", no_wrap=False)

    for ds in report.scores:
        key = ds.dimension.value
        display_name, higher_is_better, direction = _DIM_META.get(key, (key, True, "↑"))

        # Colour
        if higher_is_better is None:
            style = _verbosity_style(ds.score)
        else:
            style = score_style(ds.score, higher_is_better=higher_is_better)

        bar = make_bar(ds.score)

        score_table.add_row(
            display_name,
            f"[{style}]{direction}[/]",
            f"[{style}]{ds.score:.2f}[/]",
            f"[{style}]{bar}[/]",
            ds.interpretation,
        )

    scores_panel = Panel(
        score_table,
        title=f"[brand]Prism Scan[/]  [dim]·[/]  [bold]{report.model_id}[/]",
        border_style="bright_cyan",
        padding=(0, 0),
    )

    # ── Geometric separation ratio ────────────────────────────────────────
    gsr = report.geometric_separation_ratio
    gsr_min, gsr_max = 125.0, 1376.0
    gsr_norm = (gsr - gsr_min) / (gsr_max - gsr_min)
    gsr_bar = make_bar(gsr_norm, width=38)

    gsr_style = "score.good" if gsr_norm >= 0.5 else ("score.mid" if gsr_norm >= 0.25 else "score.bad")

    gsr_text = Text()
    gsr_text.append(f"  {gsr_bar}\n", style=gsr_style)
    gsr_text.append(f"  125×", style="dim white")
    gsr_text.append(" " * 32, style="")
    gsr_text.append("1,376×\n", style="dim white")
    gsr_text.append(f"\n  Separation ratio: ", style="info.key")
    gsr_text.append(f"{gsr:.0f}×", style=f"bold {gsr_style}")
    gsr_text.append("  (16-dim fiber space)", style="dim white")

    gsr_panel = Panel(
        gsr_text,
        title="[brand]Geometric Separation[/]  [dim]16-dim fiber space  ·  range: 125× → 1,376×[/]",
        border_style="cyan",
        padding=(0, 0),
    )

    # ── Duration footer ───────────────────────────────────────────────────
    duration_text = Text(
        f"  Scan completed in {report.scan_duration_ms:.0f} ms",
        style="dim white",
    )

    return Group(scores_panel, gsr_panel, duration_text)


# ---------------------------------------------------------------------------
# Phase stub renderer
# ---------------------------------------------------------------------------

def render_phase_stub(
    feature: str,
    phase: int,
    requires_pro: bool = True,
    description: str = "",
) -> Panel:
    """Generic 'coming in Phase N' panel for gated commands."""
    lines = Text()
    lines.append(f"  {feature} ", style="stub.phase bold")
    lines.append(f"is implemented in Phase {phase}.\n", style="dim white")
    if description:
        lines.append(f"\n  {description}\n", style="dim white")
    if requires_pro:
        lines.append("\n  Requires a ", style="dim white")
        lines.append("Pro license", style="stub.pro bold")
        lines.append(" · $20/mo · ", style="dim white")
        lines.append("buildmaxxing.com/prism/upgrade", style="stub.pro underline")

    return Panel(
        lines,
        title=f"[stub.phase]Phase {phase}[/] — [bold]{feature}[/]",
        border_style="yellow",
        padding=(0, 1),
    )
