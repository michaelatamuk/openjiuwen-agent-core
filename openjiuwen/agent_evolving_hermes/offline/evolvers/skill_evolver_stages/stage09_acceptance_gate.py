from __future__ import annotations

from pathlib import Path
from typing import Optional


def apply_acceptance_gate(
    improvement: float,
    min_improvement: float,
    evolved_text: str,
    cross_run_delta: Optional[float],
    output_dir: Path,
    console,
) -> bool:
    """Apply the min_improvement threshold gate.

    If improvement < min_improvement:
      - Saves evolved text to evolved_REGRESSION.md.
      - Prints a warning with the cross-run delta if available.
      - Returns False (rejected).

    If improvement >= min_improvement but still negative:
      - Prints a soft warning (accepted below zero because threshold allows it).
      - Returns True (accepted).

    Returns True (accepted) or False (rejected).
    """
    console.print("\n[blue]~~~ Evolving Stage 09 - Acceptance Gate Applying Started ~~~[/blue]")

    accepted = improvement >= min_improvement
    if not accepted:
        regression_path = output_dir / "evolved_REGRESSION.md"
        regression_path.write_text(evolved_text, encoding="utf-8")
        console.print(
            f"[yellow]⚠ Improvement {improvement:+.4f} < threshold {min_improvement:+.4f} "
            f"— not deploying (saved to evolved_REGRESSION.md)[/yellow]"
        )
        if cross_run_delta is not None:
            delta_color = "green" if cross_run_delta >= 0 else "red"
            console.print(
                f"[{delta_color}]Cross-run delta vs prior evolved: "
                f"{cross_run_delta:+.4f}[/{delta_color}]"
            )
    else:
        if improvement < 0:
            console.print(
                f"[yellow]⚠ Improvement {improvement:+.4f} is negative "
                f"(accepted because threshold is {min_improvement:+.4f})[/yellow]"
            )

    console.print("[blue]~~~ Evolving Stage 09 - Acceptance Gate Applying Finished ~~~[/blue]")
    return accepted
