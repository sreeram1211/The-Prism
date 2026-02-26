"""
prism resolve <model_id>

Downloads only config.json from HuggingFace Hub and displays the full
architecture descriptor with LoRA target recommendations.
"""

from __future__ import annotations

import contextlib
import json
import sys

import typer
from rich.spinner import Spinner
from rich.live import Live

from prism.cli.display.theme import make_console, print_banner
from prism.cli.display.renderers import render_resolver_result
from prism.config import get_settings
from prism.resolver.auto_resolver import (
    GatedModelError,
    ModelNotFoundError,
    ConfigParseError,
    PrismAutoResolver,
    ResolverError,
)


def resolve_command(
    model_id: str = typer.Argument(
        ...,
        help="HuggingFace model ID, e.g. 'mistralai/Mistral-7B-Instruct-v0.2'",
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
    minimal: bool = typer.Option(
        False,
        "--minimal",
        help="Show Q+V minimal LoRA targets only.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Output raw JSON instead of the pretty display.",
    ),
) -> None:
    """Resolve a HuggingFace model: detect architecture and recommend LoRA targets."""

    # Fresh console bound to the current sys.stdout (captures correctly in tests)
    console = make_console()
    settings = get_settings()

    if not as_json:
        print_banner(console)
        console.rule("[brand]Auto-Resolver[/]", style="cyan")
        console.print()

    # ── Resolve ──────────────────────────────────────────────────────────────
    resolver = PrismAutoResolver(
        hf_token=hf_token or settings.hf_token,
        cache_dir=settings.hf_cache_dir,
    )

    # Use a spinner for pretty output; skip it for --json so the captured
    # stdout contains only the JSON payload (no ANSI escape sequences).
    spinner_ctx: contextlib.AbstractContextManager = (
        Live(
            Spinner("dots", text=f"[brand.dim]Resolving[/] [bold]{model_id}[/] …"),
            console=console,
            transient=True,
            refresh_per_second=12,
        )
        if not as_json
        else contextlib.nullcontext()
    )

    with spinner_ctx:
        try:
            result = resolver.resolve(model_id, revision=revision)
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

    if as_json:
        # Write directly via console.file (the _NamedTextIOWrapper that backs
        # CliRunner's BytesIOCopy) rather than through typer.echo / sys.stdout.
        # Under pytest's capture mode Click's cached _default_text_stdout()
        # may resolve to the outer capture stream instead of CliRunner's buffer.
        console.file.write(json.dumps(result.to_dict(), indent=2) + "\n")
        console.file.flush()
        return

    console.print(render_resolver_result(result))
    console.print()
    console.print(
        f"  [ok]✓[/] Resolved [bold]{model_id}[/]  "
        f"·  family: {result.arch.family.value}  "
        f"·  ~{result.arch.param_count_estimate / 1e9:.1f}B params  "
        f"·  recommended rank: [lora.rank]{result.arch.lora_rank_recommendation}[/]"
    )
    console.print()
