from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ._base_gate import BaseAcceptanceGate


# ── Threshold gate (no Thompson Sampling) ────────────────────────────────────────────────────

class ThresholdAcceptanceGate(BaseAcceptanceGate):
    """Accept if improvement >= min_improvement — identical to the original.

    ``decide()`` always returns ``(accepted, None)``; the second element is
    always None because there is no TS confidence to report.
    """

    def __init__(self, min_improvement: float = 0.0) -> None:
        self._min = min_improvement

    def decide(self, improvement: float, evolved_score: float, skill_name: str, evolved_text: str,
               cross_run_delta: Optional[float], output_dir: Path, console) -> Tuple[bool, Optional[float]]:
        accepted = improvement >= self._min

        sign = "+" if improvement >= 0 else ""
        score_line = f"  Score change: {sign}{improvement:.4f}"

        console.print("\nAcceptance gate  (threshold only):")

        if accepted:
            if improvement < 0:
                console.print(f"{score_line}  [yellow]below zero but threshold allows ≥ {self._min:+.4f}[/yellow]")
            else:
                console.print(f"{score_line}  [green]✓ above minimum {self._min:.4f}[/green]")
            console.print("  Decision: [green]ACCEPTED — evolved skill will be deployed[/green]")
        else:
            console.print(f"{score_line}  [red]✗ below minimum {self._min:.4f}[/red]")
            console.print("  Decision: [red]REJECTED — saved to evolved_REGRESSION.md[/red]")
            (output_dir / "evolved_REGRESSION.md").write_text(evolved_text, encoding="utf-8")
            trend = self._trend_line(cross_run_delta)
            if trend:
                color = "green" if cross_run_delta >= 0 else "red"
                console.print(f"  [{color}]{trend.strip()}[/{color}]")

        return accepted, None
