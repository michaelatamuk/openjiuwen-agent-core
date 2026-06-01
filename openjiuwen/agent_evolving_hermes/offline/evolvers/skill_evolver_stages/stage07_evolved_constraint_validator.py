from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.constraints import ConstraintValidator


def validate_evolved_constraints(
    evolved_text: str,
    skill_raw: str,
    config: EvolverConfig,
    output_dir: Path,
    console,
) -> Tuple[List, bool]:
    """Validate the evolved skill text against all constraints.

    On failure: saves evolved_FAILED.md to output_dir and prints each failure.
    On success: prints confirmation.
    Always returns (checks, passed) — never raises.
    """
    validator = ConstraintValidator(config)
    checks = validator.validate_all(
        evolved_text, artifact_type="skill", baseline_text=skill_raw
    )
    failures = [c for c in checks if not c.passed]
    if failures:
        failed_path = output_dir / "evolved_FAILED.md"
        failed_path.write_text(evolved_text, encoding="utf-8")
        for f in failures:
            console.print(
                f"[red]✗ EVOLVED CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]"
            )
        console.print(f"[dim]Saved failed variant to {failed_path}[/dim]")
        return checks, False
    console.print("[green]✓ Evolved constraints passed[/green]")
    return checks, True
