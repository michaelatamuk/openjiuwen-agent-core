from __future__ import annotations

from pathlib import Path
from typing import List

from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.offline.constraints import ConstraintValidator


def validate_evolved_constraints(
    evolved_text: str,
    skill_raw: str,
    config: EvolverConfig,
    output_dir: Path,
    console,
) -> List:
    """Validate the evolved skill text against all constraints.

    On failure: saves evolved_FAILED.md to output_dir, prints each failure,
    then raises ValueError so the evolution run is rejected.
    On success: prints confirmation and returns the full checks list.
    """
    validator = ConstraintValidator(config)
    checks = validator.validate_all(
        evolved_text, artifact_type="skill", baseline_text=skill_raw
    )
    failures = [c for c in checks if not c.passed]
    if failures:
        failed_path = output_dir / "evolved_FAILED.md"
        failed_path.write_text(evolved_text)
        for f in failures:
            console.print(
                f"[red]✗ EVOLVED CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]"
            )
        console.print(f"[dim]Saved failed variant to {failed_path}[/dim]")
        raise ValueError("Evolved skill fails constraints — evolution rejected.")
    console.print("[green]✓ Evolved constraints passed[/green]")
    return checks
