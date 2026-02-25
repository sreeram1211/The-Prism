"""
prism serve

Launches the Prism FastAPI server via uvicorn.
"""

from __future__ import annotations

import typer

from prism.cli.display.theme import make_console, print_banner
from prism import __version__


def serve_command(
    host: str = typer.Option(
        "127.0.0.1",
        "--host", "-h",
        help="Host address to bind.",
    ),
    port: int = typer.Option(
        8000,
        "--port", "-p",
        help="Port number.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable hot-reload (development mode).",
    ),
    workers: int = typer.Option(
        1,
        "--workers", "-w",
        help="Number of worker processes (ignored when --reload is set).",
    ),
) -> None:
    """Start the Prism FastAPI server."""
    console = make_console()
    try:
        import uvicorn
    except ImportError:
        console.print("[error]uvicorn not installed.[/] Run: pip install 'uvicorn[standard]'")
        raise typer.Exit(1)

    print_banner(console)
    console.rule("[brand]Prism API Server[/]", style="cyan")
    console.print()
    console.print(
        f"  Version  [bold]{__version__}[/]\n"
        f"  Listening on [bold cyan]http://{host}:{port}[/]\n"
        f"  Docs  →  [bold cyan]http://{host}:{port}/docs[/]\n"
        f"  Hot-reload: {'[ok]on[/]' if reload else '[dim]off[/]'}\n"
        f"  Workers: [bold]{workers if not reload else 1}[/]"
    )
    console.print()

    uvicorn.run(
        "prism.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )
