# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unified skill constraint validator — baseline (stage 02) and evolved (stage 07)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from .constraint_result import ConstraintResult
from .constraint_validator import ConstraintValidator


def validate_skill_constraints(skill_text: str,
                               config: EvolverConfig,
                               console,
                               stage_label: str,
                               baseline_text: Optional[str] = None,
                               output_dir: Optional[Path] = None) -> Tuple[List, bool]:
    """Validate skill text against all constraints.

    Used for both the baseline check (stage 02, before evolution) and the
    evolved check (stage 07, after GEPA produces a candidate).

    Args:
        skill_text:    The skill text to validate.
        config:        EvolverConfig (constraint thresholds live here).
        console:       Rich Console for progress output.
        stage_label:   Display label used in banners and failure messages,
                       e.g. ``"Baseline"`` or ``"Evolved"``.
        baseline_text: Original baseline skill text.  When provided, enables
                       relative/delta constraints (e.g. max size growth).
                       Pass ``None`` for the baseline check itself.
        output_dir:    When provided and validation fails, the failed skill
                       text is saved as ``evolved_FAILED.md`` here.

    Returns:
        ``(checks, passed)`` — never raises.
    """
    console.print(f"\n[blue]~~~ {stage_label} Skill Constraints Validation Started ~~~[/blue]")

    validator = ConstraintValidator(config)
    kwargs = {}
    if baseline_text is not None:
        kwargs["baseline_text"] = baseline_text
    constraint_results: List[ConstraintResult] = validator.validate_all(skill_text, artifact_type="skill", **kwargs)

    failures = [c for c in constraint_results if not c.passed]
    if failures:
        if output_dir is not None:
            failed_path = output_dir / "evolved_FAILED.md"
            failed_path.write_text(skill_text, encoding="utf-8")
            console.print(f"[dim]Saved failed variant to {failed_path}[/dim]")
        for f in failures:
            console.print(f"[red]✗ {stage_label.upper()} CONSTRAINT FAILED: {f.constraint_name}: {f.message}[/red]")
        result = False
    else:
        console.print(f"[green]✓ {stage_label} — {len(constraint_results)}/{len(constraint_results)} constraints passed[/green]")
        result = True

    console.print(f"[blue]~~~ {stage_label} Skill Constraints Validation Finished ~~~[/blue]")
    return constraint_results, result
