"""
prism info

Shows system information, installed dependencies, and license status.
"""

from __future__ import annotations

import platform
import sys

import typer
from rich.table import Table

from prism import __version__
from prism.cli.display.theme import make_console, print_banner
from prism.config import get_settings
from prism.core.license import LicenseTier, get_verifier


def info_command(
    license_key: str | None = typer.Option(
        None,
        "--license-key", "-k",
        help="Prism license key to verify.",
        show_default=False,
    ),
) -> None:
    """Show system info, dependency versions, and license status."""
    console = make_console()
    print_banner(console)
    console.rule("[brand]System Info[/]", style="cyan")
    console.print()

    settings = get_settings()

    # ── Runtime table ─────────────────────────────────────────────────────
    rt = Table(show_header=False, box=None, padding=(0, 2))
    rt.add_column("Key",   style="info.key",   no_wrap=True)
    rt.add_column("Value", style="info.value")

    rt.add_row("Prism version",  f"[bold]{__version__}[/]")
    rt.add_row("Python",         f"{sys.version.split()[0]}  ({platform.python_implementation()})")
    rt.add_row("Platform",       f"{platform.system()} {platform.machine()}")

    # Installed package versions
    _add_pkg_row(rt, "huggingface-hub")
    _add_pkg_row(rt, "fastapi")
    _add_pkg_row(rt, "pydantic")
    _add_pkg_row(rt, "rich")
    _add_pkg_row(rt, "typer")

    rt.add_row("HF cache dir",   settings.hf_cache_dir)
    rt.add_row("API prefix",     settings.api_prefix)
    rt.add_row("License mode",   settings.license_mode)

    console.print(rt)
    console.print()

    # ── License check ─────────────────────────────────────────────────────
    console.rule("[brand]License[/]", style="cyan")
    console.print()

    verifier = get_verifier()
    key = license_key or ""
    info = verifier.verify(key)

    tier_style = "ok" if info.tier == LicenseTier.PRO else "warn"
    tier_label = "PRO" if info.tier == LicenseTier.PRO else "FREE"
    valid_label = "[ok]✓ Valid[/]" if info.valid else "[error]✗ Invalid[/]"

    lt = Table(show_header=False, box=None, padding=(0, 2))
    lt.add_column("Key",   style="info.key",   no_wrap=True)
    lt.add_column("Value", style="info.value")

    lt.add_row("Status",    valid_label)
    lt.add_row("Tier",      f"[{tier_style}][bold]{tier_label}[/][/]")
    lt.add_row("Message",   info.message)
    if info.expires_at:
        lt.add_row("Expires", info.expires_at)

    console.print(lt)
    console.print()

    if info.tier == LicenseTier.FREE:
        console.print(
            "  [dim]Free tier:[/]  Prism Scan is available without a license.\n"
            "  [stub.pro]Upgrade to Pro[/] [dim]for Generate, Monitor & Agent →[/] "
            "[stub.pro underline]buildmaxxing.com/prism/upgrade[/]\n"
        )

    # ── Feature gate table ────────────────────────────────────────────────
    console.rule("[brand]Feature Availability[/]", style="cyan")
    console.print()

    ft = Table(show_header=True, header_style="bold white", box=None, padding=(0, 2))
    ft.add_column("Feature",   style="info.value", no_wrap=True)
    ft.add_column("Phase",     justify="center",   no_wrap=True)
    ft.add_column("License",   justify="center",   no_wrap=True)
    ft.add_column("Status",    justify="center",   no_wrap=True)

    _add_feature_row(ft, "Prism Resolve",  1, "Free", available=True)
    _add_feature_row(ft, "Prism Scan",     2, "Free", available=True)
    _add_feature_row(ft, "Prism Generate", 3, "Pro",  available=(info.tier == LicenseTier.PRO))
    _add_feature_row(ft, "Prism Monitor",  4, "Pro",  available=(info.tier == LicenseTier.PRO))
    _add_feature_row(ft, "Prism Agent",    5, "Pro",  available=(info.tier == LicenseTier.PRO))

    console.print(ft)
    console.print()


def _add_pkg_row(table: Table, pkg: str) -> None:
    """Attempt to fetch installed package version and add it to the table."""
    try:
        from importlib.metadata import version as pkg_version
        ver = pkg_version(pkg)
        table.add_row(pkg, ver)
    except Exception:
        table.add_row(pkg, "[dim]not installed[/]")


def _add_feature_row(table: Table, name: str, phase: int, tier: str, available: bool) -> None:
    status = "[ok]✓ Available[/]" if available else "[error]✗ Requires Pro[/]"
    phase_str = f"[dim]Phase {phase}[/]"
    tier_str = "[dim]Free[/]" if tier == "Free" else "[stub.pro]Pro[/]"
    table.add_row(name, phase_str, tier_str, status)
