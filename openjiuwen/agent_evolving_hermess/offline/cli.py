# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CLI entry point for the offline GEPA skill evolver.

Mirrors hermes-agent-self-evolution evolve_skill CLI exactly, plus:
  --dry-run         Validate setup and print what would happen — no LLM calls.
  --reuse-dataset   Reuse the most recently cached dataset instead of regenerating.
  --min-improvement Minimum fitness improvement required before accepting evolved skill.
  --all             Evolve ALL skills under skills-root in one invocation.

Usage:
    python -m openjiuwen.agent_evolving_hermess.offline --skill my-skill --iterations 10
    python -m openjiuwen.agent_evolving_hermess.offline --all --min-improvement 0.05
    python -m openjiuwen.agent_evolving_hermess.offline.cli --skill my-skill --dry-run
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import click

from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.offline.evolve import batch_evolve, evolve


def _make_config(
    iterations: int,
    optimizer_model: str,
    eval_model: str,
    output_dir: str,
    run_pytest: bool,
    skills_root: Optional[str],
) -> EvolverConfig:
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
    return config


@click.command()
@click.option("--skill", default=None, help="Skill name to evolve (mutually exclusive with --all).")
@click.option(
    "--all", "evolve_all",
    is_flag=True,
    default=False,
    help="Evolve ALL non-archived skills under --skills-root in one invocation.",
)
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
@click.option(
    "--min-improvement",
    default=0.0,
    show_default=True,
    type=float,
    help=(
        "Minimum fitness improvement required before accepting the evolved skill. "
        "E.g. 0.05 requires at least +5% improvement; 0.0 (default) accepts any positive delta. "
        "Negative values (e.g. -0.02) accept up to 2% regression."
    ),
)
def main(
    skill: Optional[str],
    evolve_all: bool,
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
    min_improvement: float,
) -> None:
    """Evolve a Jiuwen SKILL.md using GEPA genetic prompt optimisation."""

    if not skill and not evolve_all:
        raise click.UsageError("Provide --skill <name> or --all.")
    if skill and evolve_all:
        raise click.UsageError("--skill and --all are mutually exclusive.")

    config = _make_config(iterations, optimizer_model, eval_model, output_dir, run_pytest, skills_root)

    # ── Resolve skill list for --all ─────────────────────────────────────────
    if evolve_all:
        from openjiuwen.agent_evolving_hermess.online.skill_store import skill_list

        skill_names = asyncio.run(skill_list(config.skills_root, include_archived=False))
        if not skill_names:
            click.echo(f"No skills found under {config.skills_root}.")
            return
        click.echo(f"Evolving {len(skill_names)} skill(s): {', '.join(skill_names)}\n")
    else:
        skill_names = [skill]  # type: ignore[list-item]

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if dry_run:
        from openjiuwen.agent_evolving_hermess.offline.constraints import ConstraintValidator
        from openjiuwen.agent_evolving_hermess.offline.skill_module import find_skill, load_skill

        any_failed = False
        for name in skill_names:
            click.echo(f"\n[DRY RUN] Validating '{name}'")
            click.echo(f"  Skills root     : {config.skills_root}")
            click.echo(f"  Eval source     : {eval_source}")
            click.echo(f"  Iterations      : {iterations}")
            click.echo(f"  Optimizer model : {optimizer_model}")
            click.echo(f"  Eval model      : {eval_model}")
            click.echo(f"  Min improvement : {min_improvement:+.4f}")

            skill_path = find_skill(name, config.skills_root)
            if skill_path is None:
                click.echo(f"  ✗ Skill '{name}' not found under {config.skills_root}")
                any_failed = True
                continue

            skill_data = load_skill(skill_path)
            validator = ConstraintValidator(config)
            checks = validator.validate_all(skill_data["raw"], artifact_type="skill")
            failures = [c for c in checks if not c.passed]

            if failures:
                for f in failures:
                    click.echo(f"  ✗ {f.constraint_name}: {f.message}")
                any_failed = True
            else:
                click.echo(f"  Skill found     : {skill_path}")
                click.echo(f"  Skill size      : {len(skill_data['raw'])} chars")
                click.echo(f"  Constraints     : ALL PASSED")
                click.echo(f"  Would generate eval dataset (source: {eval_source})")
                click.echo(f"  Would run GEPA optimization ({iterations} iterations)")
                click.echo(f"  Would apply min_improvement gate ({min_improvement:+.4f})")
                click.echo(f"  Would save results to {config.output_dir / name}")

        if any_failed:
            click.echo("\n[DRY RUN] Some validations FAILED.")
            raise SystemExit(1)
        click.echo("\n[DRY RUN] All validations passed — no LLM calls made.")
        return

    # ── Single skill ─────────────────────────────────────────────────────────
    if not evolve_all:
        metrics = evolve(
            skill_name=skill_names[0],
            eval_source=eval_source,
            external_sources=list(external_sources) if external_sources else None,
            iterations=iterations,
            config=config,
            reuse_dataset=reuse_dataset,
            min_improvement=min_improvement,
        )
        _print_summary([metrics])
        return

    # ── Batch (--all) ─────────────────────────────────────────────────────────
    all_metrics = batch_evolve(
        skill_names=skill_names,
        eval_source=eval_source,
        external_sources=list(external_sources) if external_sources else None,
        iterations=iterations,
        config=config,
        reuse_dataset=reuse_dataset,
        min_improvement=min_improvement,
    )
    _print_summary(all_metrics)


def _print_summary(results: list) -> None:
    click.echo("\n── Evolution summary ────────────────────────────────")
    for m in results:
        name = m.get("skill_name", "?")
        if "error" in m:
            click.echo(f"  {name}: ERROR — {m['error']}")
        else:
            accepted = "✓" if m.get("accepted", True) else "✗"
            click.echo(
                f"  {accepted} {name}: "
                f"baseline={m.get('baseline_score', 0):.4f} "
                f"evolved={m.get('evolved_score', 0):.4f} "
                f"improvement={m.get('improvement', 0):+.4f} "
                f"({'ACCEPTED' if m.get('accepted', True) else 'REJECTED'})"
            )
    click.echo("")


if __name__ == "__main__":
    main()
