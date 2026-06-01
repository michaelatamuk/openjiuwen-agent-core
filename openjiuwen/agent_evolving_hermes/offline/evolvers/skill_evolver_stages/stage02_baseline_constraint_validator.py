from __future__ import annotations

from typing import List

from offline.evolvers.config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.constraints import ConstraintValidator


def validate_baseline_constraints(
    skill_raw: str,
    config: EvolverConfig,
    console,
) -> List:
    """Validate the baseline skill against all constraints.

    Prints per-failure messages to console.
    Raises ValueError if any constraint fails.
    Returns the full list of ConstraintResult objects (all passed).
    """
    validator = ConstraintValidator(config)
    checks = validator.validate_all(skill_raw, artifact_type="skill")
    failures = [c for c in checks if not c.passed]
    if failures:
        for f in failures:
            console.print(
                f"[red]✗ BASELINE CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]"
            )
        raise ValueError("Baseline skill fails constraints — fix before evolving.")
    console.print("[green]✓ Baseline constraints passed[/green]")
    return checks
