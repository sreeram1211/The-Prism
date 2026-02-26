"""
Rich theme and shared brand primitives for The Prism CLI.

All command functions should call ``make_console()`` at the top of their body
to create a Console bound to the *current* sys.stdout.  This ensures that
test runners (e.g. Click's CliRunner, which temporarily replaces sys.stdout
with a BytesIO-backed wrapper) capture Rich output correctly.

The module-level ``console`` is provided as a convenience for interactive /
top-level scripts that don't need test isolation.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PRISM_THEME = Theme(
    {
        # Brand
        "brand":           "bold bright_cyan",
        "brand.dim":       "cyan",
        "brand.by":        "dim italic white",
        # Architecture families
        "family.attention":       "bold cyan",
        "family.ssm":             "bold magenta",
        "family.hybrid":          "bold yellow",
        "family.encoder_decoder": "bold blue",
        "family.unknown":         "dim white",
        # Scores
        "score.good":  "bold bright_green",
        "score.mid":   "bold yellow",
        "score.bad":   "bold red",
        "score.label": "bold white",
        # LoRA
        "lora.target":     "bold bright_cyan",
        "lora.target.min": "dim cyan",
        "lora.rank":       "bold magenta",
        # Info labels
        "info.key":   "dim white",
        "info.value": "white",
        "info.unit":  "dim white",
        # Phase stubs
        "stub.phase": "yellow",
        "stub.pro":   "bright_yellow",
        # Status
        "ok":    "bold green",
        "warn":  "bold yellow",
        "error": "bold red",
    }
)


def make_console() -> Console:
    """
    Create a themed Console bound to the *current* ``sys.stdout``.

    We capture ``sys.stdout`` explicitly at creation time so that Rich writes
    to Click's ``_NamedTextIOWrapper(BytesIOCopy)`` when running under
    ``typer.testing.CliRunner``.  (With ``file=None`` Rich defers the lookup to
    call time, which can race with ``sys.stdout`` being restored.)

    The ``BytesIOCopy`` is guarded against premature ``close()`` calls by the
    ``_SafeBytesIOCopy`` patch installed in ``tests/conftest.py``, so the
    ``TextIOWrapper.__del__`` path that would otherwise close the buffer when
    the Console is garbage-collected after the command returns is harmless in
    tests.  In production the buffer is a real file and no patching occurs.
    """
    return Console(theme=PRISM_THEME, highlight=False, file=sys.stdout)


# Module-level convenience console (used by non-test interactive code).
# Prefer make_console() inside command functions.
console = make_console()


# ---------------------------------------------------------------------------
# Family → styled string
# ---------------------------------------------------------------------------

FAMILY_STYLES: dict[str, str] = {
    "attention":       "[family.attention]⬡ Attention[/]",
    "ssm":             "[family.ssm]⬡ SSM[/]",
    "hybrid":          "[family.hybrid]⬡ Hybrid (Attn+SSM)[/]",
    "encoder_decoder": "[family.encoder_decoder]⬡ Encoder-Decoder[/]",
    "unknown":         "[family.unknown]⬡ Unknown[/]",
}


def styled_family(family: str) -> str:
    return FAMILY_STYLES.get(family, f"[family.unknown]{family}[/]")


# ---------------------------------------------------------------------------
# Score colour helper
# ---------------------------------------------------------------------------

def score_style(score: float, higher_is_better: bool = True) -> str:
    """Return a Rich style name for a 0–1 score."""
    effective = score if higher_is_better else (1.0 - score)
    if effective >= 0.65:
        return "score.good"
    if effective >= 0.40:
        return "score.mid"
    return "score.bad"


def make_bar(score: float, width: int = 22) -> str:
    """Return a Unicode block progress bar string."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Header / banner
# ---------------------------------------------------------------------------

BANNER = (
    "[brand]  ██████╗ ██████╗ ██╗███████╗███╗   ███╗[/]\n"
    "[brand] ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║[/]\n"
    "[brand] ██████╔╝██████╔╝██║███████╗██╔████╔██║[/]\n"
    "[brand] ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║[/]\n"
    "[brand] ██║     ██║  ██║██║███████║██║ ╚═╝ ██║[/]\n"
    "[brand] ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝[/]\n"
    "[brand.by]     The Prism · V1.0 · by BuildMaxxing[/]"
)


def print_banner(c: Console | None = None) -> None:
    """Print the ASCII banner. Pass a console or use the module-level one."""
    out = c or console
    out.print()
    out.print(BANNER, justify="center")
    out.print()
