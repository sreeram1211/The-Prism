"""
The Prism CLI — root Typer application.

Entry point:  prism  (registered in pyproject.toml [project.scripts])

Commands
────────
  prism resolve <model_id>   Detect architecture + recommend LoRA targets
  prism scan    <model_id>   9-dimensional behavioral diagnostic (Phase 2 mock)
  prism generate <model_id>  Compile LoRA adapter [Phase 3, Pro]
  prism monitor  <model_id>  Real-time telemetry stream [Phase 4, Pro]
  prism agent                Autonomous chat with vector memory [Phase 5, Pro]
  prism serve                Start the FastAPI server
  prism info                 System info & license status
"""

from __future__ import annotations

import typer

from prism import __version__
from prism.cli.commands.cmd_resolve  import resolve_command
from prism.cli.commands.cmd_scan     import scan_command
from prism.cli.commands.cmd_generate import generate_command, monitor_command, agent_command
from prism.cli.commands.cmd_serve    import serve_command
from prism.cli.commands.cmd_info     import info_command
from prism.cli.commands.cmd_history  import history_command, diff_command

# ---------------------------------------------------------------------------
# Root application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="prism",
    help=(
        "[bold bright_cyan]The Prism[/bold bright_cyan] — "
        "Local-first AI behavioral manifold tooling suite by BuildMaxxing.\n\n"
        "Maps, monitors, and manipulates LLM behaviour using a proprietary "
        "16-dimensional fiber space."
    ),
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,  # we handle our own error formatting
)

# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

app.command("resolve",  help="Detect architecture + recommend LoRA targets.")(resolve_command)
app.command("scan",     help="9-dimensional behavioral diagnostic.")(scan_command)
app.command("generate", help="[Phase 3, Pro] Compile a precision LoRA adapter.")(generate_command)
app.command("monitor",  help="[Phase 4, Pro] Real-time activation telemetry.")(monitor_command)
app.command("agent",    help="[Phase 5, Pro] Autonomous chat with vector memory.")(agent_command)
app.command("serve",    help="Start the Prism FastAPI server.")(serve_command)
app.command("info",     help="System info, dependency versions & license status.")(info_command)
app.command("history",  help="[Phase 6] List recent behavioral scans.")(history_command)
app.command("diff",     help="[Phase 6] Compare two model behavioral fingerprints.")(diff_command)

# ---------------------------------------------------------------------------
# Version flag
# ---------------------------------------------------------------------------

def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"The Prism v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version", "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """The Prism — Local-first AI behavioral manifold tooling by BuildMaxxing."""


if __name__ == "__main__":
    app()
