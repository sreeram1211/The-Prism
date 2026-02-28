"""
prism history  — list recent scans in a Rich table
prism diff     — compare two model scans side-by-side

Phase 6: queries the local SQLite DB directly (no server required).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import timezone

import typer
from rich.table import Table

from prism.cli.display.theme import make_console
from prism.db.engine import SessionLocal
from prism.db.migrations import create_all
from prism.db.models import ScanRecord


def _ensure_db() -> None:
    """Create tables if they don't exist yet (first-run protection)."""
    create_all()


def _score_bar(score: float) -> str:
    """Return a compact 5-char Unicode bar for a 0–1 score."""
    filled = round(score * 5)
    return "█" * filled + "░" * (5 - filled)


# ---------------------------------------------------------------------------
# prism history
# ---------------------------------------------------------------------------

def history_command(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of scans to display."),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Filter by model ID (substring match)."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """
    List recent behavioral scans stored in the local Prism database.
    """
    # Capture stdout BEFORE _ensure_db() so pytest live-log capture
    # suspend/resume cannot overwrite sys.stdout with the capture proxy.
    _out = sys.stdout
    _ensure_db()
    console = make_console()

    with SessionLocal() as db:
        q = db.query(ScanRecord)
        if model:
            q = q.filter(ScanRecord.model_id.contains(model))
        records = q.order_by(ScanRecord.created_at.desc()).limit(limit).all()

    if as_json:
        out = []
        for r in records:
            scores = json.loads(r.scores_json)
            out.append({
                "scan_id": r.id,
                "model_id": r.model_id,
                "created_at": r.created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "duration_ms": r.duration_ms,
                "geo_ratio": r.geo_ratio,
                "scores": scores,
            })
        _out.write(json.dumps(out, indent=2) + "\n")
        _out.flush()
        return

    if not records:
        console.print("[dim]No scans found.[/]  Run [bold]prism scan <model_id>[/] first.")
        return

    table = Table(
        title="Prism Scan History",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Scan ID", style="dim", width=10)
    table.add_column("Model", style="bold", max_width=36)
    table.add_column("Date", style="white", width=18)
    table.add_column("Geo-ratio", justify="right")
    table.add_column("Top Dim", justify="left")
    table.add_column("Score", justify="left")

    for r in records:
        scores = json.loads(r.scores_json)
        top = max(scores, key=lambda s: s["score"])
        date_str = r.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
        table.add_row(
            r.id[:8] + "…",
            r.model_id,
            date_str,
            f"{r.geo_ratio:.0f}×",
            top["dimension"],
            _score_bar(top["score"]) + f"  {top['score']:.2f}",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# prism diff
# ---------------------------------------------------------------------------

def diff_command(
    model_a: str = typer.Argument(..., help="Model ID or scan ID (model A)"),
    model_b: str = typer.Argument(..., help="Model ID or scan ID (model B)"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """
    Compare two model behavioral fingerprints side-by-side.

    Pass either model IDs (uses the most recent scan for each) or full scan UUIDs.
    """
    # Capture stdout BEFORE _ensure_db() so pytest live-log capture
    # suspend/resume cannot overwrite sys.stdout with the capture proxy.
    _out = sys.stdout
    _ensure_db()
    console = make_console()

    def _resolve_scan(arg: str) -> ScanRecord:
        with SessionLocal() as db:
            # Try exact scan ID first
            rec = db.get(ScanRecord, arg)
            if rec:
                return rec
            # Fall back to most recent scan for model_id substring
            rec = (
                db.query(ScanRecord)
                .filter(ScanRecord.model_id.contains(arg))
                .order_by(ScanRecord.created_at.desc())
                .first()
            )
            if rec:
                return rec
        raise typer.BadParameter(
            f"No scan found for '{arg}'. Run [bold]prism scan {arg}[/] first.",
            param_hint="model_a / model_b",
        )

    rec_a = _resolve_scan(model_a)
    rec_b = _resolve_scan(model_b)

    scores_a = {s["dimension"]: s["score"] for s in json.loads(rec_a.scores_json)}
    scores_b = {s["dimension"]: s["score"] for s in json.loads(rec_b.scores_json)}

    all_dims = sorted(set(scores_a) | set(scores_b))

    deltas = []
    sq_sum = 0.0
    for dim in all_dims:
        sa = scores_a.get(dim, 0.0)
        sb = scores_b.get(dim, 0.0)
        d = round(sb - sa, 4)
        sq_sum += d ** 2
        deltas.append((dim, sa, sb, d))

    composite = math.sqrt(sq_sum)
    mean_a = sum(scores_a.values()) / len(scores_a) if scores_a else 0.0
    mean_b = sum(scores_b.values()) / len(scores_b) if scores_b else 0.0
    if abs(mean_a - mean_b) < 0.005:
        winner = "tie"
        winner_label = "Tie"
    elif mean_b > mean_a:
        winner = "b"
        winner_label = rec_b.model_id
    else:
        winner = "a"
        winner_label = rec_a.model_id

    if as_json:
        out = {
            "scan_a": rec_a.id,
            "scan_b": rec_b.id,
            "model_a": rec_a.model_id,
            "model_b": rec_b.model_id,
            "composite_distance": round(composite, 4),
            "winner": winner,
            "deltas": [
                {"dimension": d, "score_a": sa, "score_b": sb, "delta": dv}
                for d, sa, sb, dv in deltas
            ],
        }
        _out.write(json.dumps(out, indent=2) + "\n")
        _out.flush()
        return

    # Shorten model names for display
    def _short(m: str) -> str:
        return m.split("/")[-1][:20] if "/" in m else m[:20]

    name_a = _short(rec_a.model_id)
    name_b = _short(rec_b.model_id)

    console.rule(
        f"[bold cyan]Behavioral Diff[/]  ·  [bold]{name_a}[/]  vs  [bold]{name_b}[/]"
    )
    console.print()

    table = Table(show_header=True, header_style="bold", border_style="dim")
    table.add_column("Dimension", style="bold", width=14)
    table.add_column(name_a, justify="right", width=10)
    table.add_column(name_b, justify="right", width=10)
    table.add_column("Delta", justify="right", width=10)
    table.add_column("", width=2)

    for dim, sa, sb, dv in deltas:
        if abs(dv) < 0.01:
            arrow = "≈"
            style = "dim"
        elif dv > 0:
            arrow = "↑"
            style = "green"
        else:
            arrow = "↓"
            style = "red"
        table.add_row(
            dim,
            f"{sa:.3f}",
            f"{sb:.3f}",
            f"{dv:+.3f}",
            f"[{style}]{arrow}[/{style}]",
        )

    console.print(table)
    console.print()
    console.print(
        f"  Composite distance: [bold]{composite:.4f}[/]     "
        f"Winner: [bold cyan]{winner_label}[/]"
    )
    console.print()
