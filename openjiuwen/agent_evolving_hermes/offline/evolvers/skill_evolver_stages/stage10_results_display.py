from __future__ import annotations

from typing import List, Optional

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
    constraint_checks: Optional[List] = None,
) -> None:
    """Print a Rich table (or plain-text fallback) with the evolution results.

    *constraint_checks* may be either a list of ConstraintResult objects
    (with .constraint_name / .passed / .message attributes) or plain dicts
    with keys 'name', 'passed', 'message'.
    """
    def _name(c) -> str:
        return c.constraint_name if hasattr(c, "constraint_name") else c["name"]

    def _passed(c) -> bool:
        return c.passed if hasattr(c, "passed") else c["passed"]

    def _msg(c) -> str:
        return c.message if hasattr(c, "message") else c["message"]

    if _RICH:
        print()
        table = Table(
            title="Evolution Results", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("Skill", skill_name)
        table.add_row("Optimizer", optimizer_name)
        table.add_row("Iterations", str(iterations))
        table.add_row("Pre-train score", f"{baseline_score:.4f}")
        table.add_row("Evolved score", f"{evolved_score:.4f}")

        sign = "+" if improvement >= 0 else ""
        color = "green" if improvement >= 0 else "red"
        table.add_row("Improvement", f"[{color}]{sign}{improvement:.4f}[/{color}]")

        if cross_run_delta is not None:
            xr_color = "green" if cross_run_delta >= 0 else "red"
            table.add_row(
                "vs prior run",
                f"[{xr_color}]{cross_run_delta:+.4f}[/{xr_color}]",
            )

        table.add_row(
            "Accepted", "[green]✓ YES[/green]" if accepted else "[red]✗ NO[/red]"
        )
        table.add_row("Elapsed", f"{elapsed:.1f}s")
        table.add_row("Baseline chars", str(baseline_chars))
        table.add_row("Evolved chars", str(evolved_chars))

        if constraint_checks:
            n_pass = sum(1 for c in constraint_checks if _passed(c))
            n_total = len(constraint_checks)
            table.add_row("", "")  # visual separator
            summary_color = "green" if n_pass == n_total else "red"
            table.add_row(
                "Constraints",
                f"[{summary_color}]{n_pass}/{n_total} passed[/{summary_color}]",
            )
            for c in constraint_checks:
                icon = "✓" if _passed(c) else "✗"
                clr  = "green" if _passed(c) else "red"
                table.add_row(
                    f"  {_name(c)}",
                    f"[{clr}]{icon} {_msg(c)}[/{clr}]",
                )

        console.print(table)
    else:
        console.print(
            f"\nHoldout: pre-train={baseline_score:.3f} "
            f"evolved={evolved_score:.3f} "
            f"improvement={improvement:+.3f} "
            f"accepted={'YES' if accepted else 'NO'}"
        )
        if cross_run_delta is not None:
            console.print(f"vs prior run: {cross_run_delta:+.4f}")
        if constraint_checks:
            for c in constraint_checks:
                icon = "✓" if _passed(c) else "✗"
                console.print(f"  {icon} {_name(c)}: {_msg(c)}")
