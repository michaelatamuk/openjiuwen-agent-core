from __future__ import annotations

from typing import Optional

try:
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def display_results_table(
    skill_name: str,
    optimizer_name: str,
    iterations: int,
    baseline_score: float,
    evolved_score: float,
    improvement: float,
    cross_run_delta: Optional[float],
    accepted: bool,
    elapsed: float,
    baseline_chars: int,
    evolved_chars: int,
    console,
) -> None:
    """Print a Rich table (or plain-text fallback) with the evolution results."""
    if _RICH:
        table = Table(
            title="Evolution Results", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Skill", skill_name)
        table.add_row("Optimizer", optimizer_name)
        table.add_row("Iterations", str(iterations))
        table.add_row("Baseline score", f"{baseline_score:.4f}")
        table.add_row("Evolved score", f"{evolved_score:.4f}")

        sign = "+" if improvement >= 0 else ""
        color = "green" if improvement >= 0 else "red"
        table.add_row("Improvement", f"[{color}]{sign}{improvement:.4f}[/{color}]")

        if cross_run_delta is not None:
            xr_color = "green" if cross_run_delta >= 0 else "red"
            table.add_row(
                "Cross-run delta",
                f"[{xr_color}]{cross_run_delta:+.4f}[/{xr_color}]",
            )

        table.add_row(
            "Accepted", "[green]YES[/green]" if accepted else "[red]NO[/red]"
        )
        table.add_row("Elapsed", f"{elapsed:.1f}s")
        table.add_row("Baseline chars", str(baseline_chars))
        table.add_row("Evolved chars", str(evolved_chars))
        console.print(table)
    else:
        console.print(
            f"\nHoldout: baseline={baseline_score:.3f} "
            f"evolved={evolved_score:.3f} "
            f"improvement={improvement:+.3f} "
            f"accepted={'YES' if accepted else 'NO'}"
        )
        if cross_run_delta is not None:
            console.print(f"Cross-run delta: {cross_run_delta:+.4f}")
