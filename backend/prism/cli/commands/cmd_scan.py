"""
prism scan <model_id>

Runs a behavioral diagnostic across 9 dimensions and displays the results
in a colour-coded Rich table with Unicode bar charts.

Phase 2: uses MockPrismScanEngine (no model weights required).
Phase 3+: the real C++/Python probe engine will replace the mock.
"""

from __future__ import annotations

import contextlib
import json
import time

import typer
from rich.live import Live
from rich.spinner import Spinner

from prism.cli.display.theme import make_console, print_banner
from prism.cli.display.renderers import render_scan_result
from prism.config import get_settings
from prism.resolver.auto_resolver import (
    ConfigParseError,
    GatedModelError,
    ModelNotFoundError,
    PrismAutoResolver,
    ResolverError,
)
from prism.scan.engine import ScanDimension, get_scan_engine


def scan_command(
    model_id: str = typer.Argument(
        ...,
        help="HuggingFace model ID to scan, e.g. 'meta-llama/Meta-Llama-3-8B'",
        metavar="MODEL_ID",
    ),
    revision: str = typer.Option(
        "main",
        "--revision", "-r",
        help="HuggingFace revision/branch/tag.",
    ),
    hf_token: str | None = typer.Option(
        None,
        "--hf-token",
        envvar="HF_TOKEN",
        help="HuggingFace access token (for gated models).",
        show_default=False,
    ),
    dimensions: list[str] | None = typer.Option(
        None,
        "--dim", "-d",
        help=(
            "Dimensions to scan (repeat for multiple). "
            "Choices: sycophancy hedging calibration depth coherence "
            "focus specificity verbosity repetition. Default: all."
        ),
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON instead of the pretty display.",
    ),
) -> None:
    """
    Run a 9-dimensional behavioral diagnostic scan on a model.

    Downloads only config.json from HuggingFace Hub (no weights needed
    in Phase 2). The real C++/Python probe engine is wired in Phase 3.
    """
    # Fresh console bound to the current sys.stdout (captures correctly in tests)
    console = make_console()
    settings = get_settings()

    if not as_json:
        print_banner(console)
        console.rule("[brand]Prism Scan[/]  ·  Behavioral Diagnostic", style="cyan")
        console.print()

    # ── Parse requested dimensions ────────────────────────────────────────
    selected_dims: list[ScanDimension] | None = None
    if dimensions:
        valid_keys = {d.value for d in ScanDimension}
        selected_dims = []
        for raw in dimensions:
            key = raw.lower()
            if key not in valid_keys:
                console.print(
                    f"[error]Unknown dimension:[/] [bold]{raw}[/]. "
                    f"Choose from: {', '.join(sorted(valid_keys))}"
                )
                raise typer.Exit(1)
            selected_dims.append(ScanDimension(key))

    # ── Step 1: Resolve architecture (real, config-only) ─────────────────
    resolver = PrismAutoResolver(
        hf_token=hf_token or settings.hf_token,
        cache_dir=settings.hf_cache_dir,
    )

    # Spinner only for pretty output — skip for --json to keep stdout clean.
    resolve_ctx: contextlib.AbstractContextManager = (
        Live(
            Spinner("dots", text=f"[brand.dim]Resolving architecture[/] [bold]{model_id}[/] …"),
            console=console,
            transient=True,
            refresh_per_second=12,
        )
        if not as_json
        else contextlib.nullcontext()
    )

    with resolve_ctx:
        try:
            resolver_result = resolver.resolve(model_id, revision=revision)
        except ModelNotFoundError as exc:
            console.print(f"[error]✗ Model not found:[/] {exc}")
            raise typer.Exit(1)
        except GatedModelError as exc:
            console.print(f"[error]✗ Gated model:[/] {exc}")
            console.print("[dim]Tip: pass --hf-token or set the HF_TOKEN environment variable.[/]")
            raise typer.Exit(1)
        except ConfigParseError as exc:
            console.print(f"[error]✗ Config error:[/] {exc}")
            raise typer.Exit(1)
        except ResolverError as exc:
            console.print(f"[error]✗ Resolver error:[/] {exc}")
            raise typer.Exit(1)

    family = resolver_result.arch.family.value

    if not as_json:
        console.print(
            f"  [ok]✓[/] Resolved  "
            f"[bold]{model_id}[/]  ·  "
            f"family: [bold]{family}[/]  ·  "
            f"~{resolver_result.arch.param_count_estimate / 1e9:.1f}B params"
        )
        console.print()

    # ── Step 2: Run the scan (mock in Phase 2) ────────────────────────────
    engine = get_scan_engine(mock=True, family=family)

    scan_ctx: contextlib.AbstractContextManager = (
        Live(
            Spinner(
                "aesthetic",
                text="[brand.dim]Running behavioral probes[/] across 9 dimensions …",
            ),
            console=console,
            transient=True,
            refresh_per_second=12,
        )
        if not as_json
        else contextlib.nullcontext()
    )

    with scan_ctx:
        # Small artificial delay so the spinner is visible in demos
        # The real engine will take seconds to minutes here.
        t0 = time.perf_counter()
        report = engine.scan(model_id, dimensions=selected_dims)
        elapsed = time.perf_counter() - t0
        # Pad the reported duration to look like real probe work (min 400ms)
        if report.scan_duration_ms < 400:
            report.scan_duration_ms = round(400 + elapsed * 1000, 2)

    if as_json:
        output = {
            "model_id": report.model_id,
            "geometric_separation_ratio": report.geometric_separation_ratio,
            "scan_duration_ms": report.scan_duration_ms,
            "scores": [
                {
                    "dimension": ds.dimension.value,
                    "score": ds.score,
                    "interpretation": ds.interpretation,
                }
                for ds in report.scores
            ],
        }
        # Write directly via console.file (same reason as cmd_resolve.py)
        console.file.write(json.dumps(output, indent=2) + "\n")
        console.file.flush()
        return

    # ── Render ────────────────────────────────────────────────────────────
    console.print(render_scan_result(report))
    console.print()
    console.print(
        "  [dim]Note: Phase 2 uses a seeded mock probe engine.[/] "
        "[dim]Real ROC-AUC-verified scores arrive in Phase 3.[/]"
    )
    console.print()
