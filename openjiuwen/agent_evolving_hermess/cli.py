# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CLI entry point for the offline GEPA skill evolver.

Mirrors hermes-agent-self-evolution evolve_skill CLI exactly, plus:
  --dry-run    Validate setup and print what would happen — no LLM calls.
  --reuse-dataset  Reuse the most recently cached dataset instead of regenerating.

Usage:
    python -m openjiuwen.agent_evolving_hermess --skill my-skill --iterations 10
    python -m openjiuwen.agent_evolving_hermess.cli --skill my-skill --dry-run
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from openjiuwen.agent_evolving_hermess.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.evolve import evolve


@click.command()
@click.option("--skill", required=True, help="Skill name to evolve.")
@click.option(
    "--iterations",
    default=10,
    show_default=True,
    help="Number of GEPA optimisation steps.",
)
@click.option(
    "--eval-source",
    type=click.Choice(["synthetic", "external", "golden"]),
    default="synthetic",
    show_default=True,
    help="Where to source the evaluation dataset from.",
)
@click.option(
    "--external-sources",
    multiple=True,
    default=["jiuwen", "claude-code"],
    help="Which external log sources to use (when --eval-source=external).",
)
@click.option(
    "--skills-root",
    type=click.Path(),
    default=None,
    help="Override skills root directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./skill_evolver_output",
    show_default=True,
    help="Where to write evolved skill and metrics.",
)
@click.option(
    "--optimizer-model",
    default="openai/gpt-4.1",
    show_default=True,
    help="LLM model used by GEPA for reflections.",
)
@click.option(
    "--eval-model",
    default="openai/gpt-4.1-mini",
    show_default=True,
    help="LLM model used for LLM-as-judge scoring.",
)
@click.option(
    "--run-pytest",
    is_flag=True,
    default=False,
    help="Run pytest after evolution (slow).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help=(
        "Validate the setup and print what would happen, "
        "but do NOT generate a dataset or call the LLM optimizer. "
        "Mirrors Hermess evolve_skill --dry-run behavior."
    ),
)
@click.option(
    "--reuse-dataset",
    is_flag=True,
    default=False,
    help="Reuse the most recently cached dataset instead of regenerating (saves LLM cost).",
)
def main(
    skill: str,
    iterations: int,
    eval_source: str,
    external_sources: tuple,
    skills_root: Optional[str],
    output_dir: str,
    optimizer_model: str,
    eval_model: str,
    run_pytest: bool,
    dry_run: bool,
    reuse_dataset: bool,
) -> None:
    """Evolve a Jiuwen SKILL.md using GEPA genetic prompt optimisation."""
    config = EvolverConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=optimizer_model,
        output_dir=Path(output_dir),
        run_pytest=run_pytest,
    )
    if skills_root:
        config.skills_root = Path(skills_root)

    # ── Dry-run mode: validate without LLM calls ─────────────────────────────
    # Mirrors Hermess evolve_skill.py lines 73-78
    if dry_run:
        from openjiuwen.agent_evolving_hermess.skill_module import find_skill
        from openjiuwen.agent_evolving_hermess.constraints import ConstraintValidator
        from openjiuwen.agent_evolving_hermess.skill_module import load_skill

        click.echo(f"\n[DRY RUN] Validating setup for skill '{skill}'")
        click.echo(f"  Skills root     : {config.skills_root}")
        click.echo(f"  Eval source     : {eval_source}")
        click.echo(f"  Iterations      : {iterations}")
        click.echo(f"  Optimizer model : {optimizer_model}")
        click.echo(f"  Eval model      : {eval_model}")

        skill_path = find_skill(skill, config.skills_root)
        if skill_path is None:
            click.echo(f"\n[red]✗ Skill '{skill}' not found under {config.skills_root}[/red]")
            raise SystemExit(1)

        skill_data = load_skill(skill_path)
        validator = ConstraintValidator(config)
        checks = validator.validate_all(skill_data["raw"], artifact_type="skill")
        failures = [c for c in checks if not c.passed]

        if failures:
            for f in failures:
                click.echo(f"  ✗ {f.constraint_name}: {f.message}")
            click.echo("\n[DRY RUN] Baseline constraints FAILED.")
            raise SystemExit(1)

        click.echo(f"\n  Skill found     : {skill_path}")
        click.echo(f"  Skill size      : {len(skill_data['raw'])} chars")
        click.echo(f"  Constraints     : ALL PASSED")
        click.echo(f"\n  Would generate eval dataset (source: {eval_source})")
        click.echo(f"  Would run GEPA optimization ({iterations} iterations)")
        click.echo(f"  Would validate constraints and save results to {config.output_dir / skill}")
        click.echo("\n[DRY RUN] Setup validated successfully — no LLM calls made.")
        return

    # ── Normal evolution run ─────────────────────────────────────────────────
    metrics = evolve(
        skill_name=skill,
        eval_source=eval_source,
        external_sources=list(external_sources) if external_sources else None,
        iterations=iterations,
        config=config,
        reuse_dataset=reuse_dataset,
    )

    click.echo("\n── Evolution complete ──────────────────────────────")
    click.echo(f"  Baseline score  : {metrics['baseline_score']:.4f}")
    click.echo(f"  Evolved score   : {metrics['evolved_score']:.4f}")
    click.echo(f"  Improvement     : {metrics['improvement']:+.4f}")
    click.echo(f"  Output dir      : {config.output_dir / skill}")


if __name__ == "__main__":
    main()
