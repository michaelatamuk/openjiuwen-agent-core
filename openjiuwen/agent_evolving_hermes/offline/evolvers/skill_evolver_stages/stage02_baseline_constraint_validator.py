from __future__ import annotations

from typing import List

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.constraints import ConstraintValidator


def validate_baseline_constraints(skill_raw: str,
                                  config: EvolverConfig,
                                  console):
    """Validate the baseline skill against all constraints.

    Prints per-failure messages to console.
    Raises ValueError if any constraint fails.
    Returns the full list of ConstraintResult objects (all passed).
    """
    console.print("\n[blue]~~~ Evolving Stage 02 - Skill Baseline Constraints Validation Started ~~~[/blue]")

    validator = ConstraintValidator(config)
    checks = validator.validate_all(skill_raw, artifact_type="skill")
    result: bool

    failures = [c for c in checks if not c.passed]
    if failures:
        for f in failures:
            console.print(f"[red]✗ BASELINE CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]")
        result = False
    else:
        console.print(f"[green]✓ Baseline — {len(checks)}/{len(checks)} constraints passed[/green]")
        result = True

    console.print("[blue]~~~ Evolving Stage 02 - Skill Baseline Constraints Validation Finished ~~~[/blue]")

    return checks, result
