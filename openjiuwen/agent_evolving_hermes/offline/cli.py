# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CLI entry point for the offline GEPA skill evolver.

Mirrors hermes-agent-self-evolution evolve_skill CLI exactly, plus:
  --dry-run         Validate setup and print what would happen — no LLM calls.
  --reuse-dataset   Reuse the most recently cached dataset instead of regenerating.
  --min-improvement Minimum fitness improvement required before accepting evolved skill.
  --all             Evolve ALL skills under skills-root in one invocation.

Usage:
    python -m openjiuwen.agent_evolving_hermes.offline --skill my-skill --iterations 10
    python -m openjiuwen.agent_evolving_hermes.offline --all --min-improvement 0.05
    python -m openjiuwen.agent_evolving_hermes.offline.cli --skill my-skill --dry-run
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import click

from offline.evolvers.skill_evolver_single_params import SkillEvolverParams
from openjiuwen.agent_evolving_hermes.offline import evolve_single_skill, evolve_skills_batch
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_prereqs import build_evolution_prereqs


def _make_config(
    iterations: int,
    optimizer_model: str,
    eval_model: str,
    output_dir: str,
    run_pytest: bool,
    skills_root: Optional[str],
    trajectory_dir: Optional[str] = None,
    trajectory_min_reward: float = 0.0,
    ts_skill_scheduler: bool = False,
    ts_example_selector: bool = False,
    ts_example_batch_size: int = 0,
    ts_acceptance_gate: bool = False,
    ts_acceptance_confidence: float = 0.75,
    ts_acceptance_n_samples: int = 100,
    ts_state_dir: Optional[str] = None,
) -> EvolverConfig:
    config = EvolverConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=optimizer_model,
        output_dir=Path(output_dir),
        run_pytest=run_pytest,
        trajectory_min_reward=trajectory_min_reward,
        ts_skill_scheduler=ts_skill_scheduler,
        ts_example_selector=ts_example_selector,
        ts_example_batch_size=ts_example_batch_size,
        ts_acceptance_gate=ts_acceptance_gate,
        ts_acceptance_confidence=ts_acceptance_confidence,
        ts_acceptance_n_samples=ts_acceptance_n_samples,
    )
    if skills_root:
        config.skills_root = Path(skills_root)
    if trajectory_dir:
        config.trajectory_dir = Path(trajectory_dir)
    if ts_state_dir:
        config.ts_state_dir = Path(ts_state_dir)
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
    type=click.Choice(["synthetic", "external", "trajectory", "golden"]),
    default="synthetic",
    show_default=True,
    help="Where to source the evaluation dataset from.",
)
@click.option(
    "--trajectory-dir",
    type=click.Path(),
    default=None,
    help="Folder of saved trajectory JSON files (when --eval-source=trajectory).",
)
@click.option(
    "--trajectory-min-reward",
    default=0.0,
    show_default=True,
    type=float,
    help="Skip trajectory steps with reward below this value (when --eval-source=trajectory).",
)
@click.option(
    "--external-sources",
    multiple=True,
    default=["jiuwen", "claude-code"],
    help=(
        "Which external log sources to use (when --eval-source=external). "
        "Choices: jiuwen, claude-code, hermes, copilot. "
        "Repeat the flag to combine sources: --external-sources jiuwen --external-sources copilot"
    ),
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
        "Mirrors Hermes evolve_skill --dry-run behavior."
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
# ── Thompson Sampling flags ───────────────────────────────────────────────────
@click.option(
    "--ts-skill-scheduler",
    is_flag=True,
    default=False,
    help=(
        "[Level 1 TS] Order skills in --all batches by Thompson Sampling priority "
        "instead of round-robin. Skills with stronger improvement history run first."
    ),
)
@click.option(
    "--ts-example-selector",
    is_flag=True,
    default=False,
    help=(
        "[Level 2 TS] Select training examples via Thompson Sampling instead of "
        "using the full trainset. Concentrates GEPA on the most discriminating examples."
    ),
)
@click.option(
    "--ts-example-batch-size",
    default=0,
    show_default=True,
    type=int,
    help=(
        "[Level 2 TS] Number of examples to select per GEPA run (requires --ts-example-selector). "
        "0 (default) = full trainset (TS ordering only, no reduction)."
    ),
)
@click.option(
    "--ts-acceptance-gate",
    is_flag=True,
    default=False,
    help=(
        "[Level 3 TS] Replace the hard improvement threshold with a Thompson Sampling gate "
        "that also requires P(candidate > deployed) >= --ts-acceptance-confidence."
    ),
)
@click.option(
    "--ts-acceptance-confidence",
    default=0.75,
    show_default=True,
    type=float,
    help=(
        "[Level 3 TS] Minimum win-probability required to accept a candidate "
        "(requires --ts-acceptance-gate). Default 0.75 = 75%% confidence."
    ),
)
@click.option(
    "--ts-acceptance-n-samples",
    default=100,
    show_default=True,
    type=int,
    help=(
        "[Level 3 TS] Monte Carlo draws for TS confidence estimate "
        "(requires --ts-acceptance-gate). Higher = more accurate but slower."
    ),
)
@click.option(
    "--ts-state-dir",
    type=click.Path(),
    default=None,
    help=(
        "Directory to persist Thompson Sampling arm state across CLI invocations. "
        "Defaults to skills-root for Level 1 and output-dir for Levels 2/3."
    ),
)
def main(
    skill: Optional[str],
    evolve_all: bool,
    iterations: int,
    eval_source: str,
    external_sources: tuple,
    trajectory_dir: Optional[str],
    trajectory_min_reward: float,
    skills_root: Optional[str],
    output_dir: str,
    optimizer_model: str,
    eval_model: str,
    run_pytest: bool,
    dry_run: bool,
    reuse_dataset: bool,
    min_improvement: float,
    ts_skill_scheduler: bool,
    ts_example_selector: bool,
    ts_example_batch_size: int,
    ts_acceptance_gate: bool,
    ts_acceptance_confidence: float,
    ts_acceptance_n_samples: int,
    ts_state_dir: Optional[str],
) -> None:
    """Evolve a Jiuwen SKILL.md using GEPA genetic prompt optimisation."""

    if not skill and not evolve_all:
        raise click.UsageError("Provide --skill <name> or --all.")
    if skill and evolve_all:
        raise click.UsageError("--skill and --all are mutually exclusive.")

    config = _make_config(
        iterations, optimizer_model, eval_model, output_dir, run_pytest, skills_root,
        trajectory_dir=trajectory_dir,
        trajectory_min_reward=trajectory_min_reward,
        ts_skill_scheduler=ts_skill_scheduler,
        ts_example_selector=ts_example_selector,
        ts_example_batch_size=ts_example_batch_size,
        ts_acceptance_gate=ts_acceptance_gate,
        ts_acceptance_confidence=ts_acceptance_confidence,
        ts_acceptance_n_samples=ts_acceptance_n_samples,
        ts_state_dir=ts_state_dir,
    )

    # ── Resolve skill list for --all ─────────────────────────────────────────
    if evolve_all:
        from online.stores.skill import skill_list

        skill_names = asyncio.run(skill_list(config.skills_root, include_archived=False))
        if not skill_names:
            click.echo(f"No skills found under {config.skills_root}.")
            return
        click.echo(f"Evolving {len(skill_names)} skill(s): {', '.join(skill_names)}\n")
    else:
        skill_names = [skill]  # type: ignore[list-item]

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if dry_run:
        from openjiuwen.agent_evolving_hermes.offline.constraints import ConstraintValidator
        from openjiuwen.agent_evolving_hermes.offline.skills.skill_module import find_skill, load_skill

        any_failed = False
        for name in skill_names:
            click.echo(f"\n[DRY RUN] Validating '{name}'")
            click.echo(f"  Skills root     : {config.skills_root}")
            click.echo(f"  Eval source     : {eval_source}")
            click.echo(f"  Iterations      : {iterations}")
            click.echo(f"  Optimizer model : {optimizer_model}")
            click.echo(f"  Eval model      : {eval_model}")
            click.echo(f"  Min improvement : {min_improvement:+.4f}")
            click.echo(f"  TS skill-sched  : {'ON' if config.ts_skill_scheduler else 'off'}")
            click.echo(f"  TS-TrainSelector: {'ON' if config.ts_example_selector else 'off'}"
                       + (f" (batch={config.ts_example_batch_size or 'all'})" if config.ts_example_selector else ""))
            click.echo(f"  TS-AcceptGate   : {'ON' if config.ts_acceptance_gate else 'off'}"
                       + (f" (conf={config.ts_acceptance_confidence:.0%})" if config.ts_acceptance_gate else ""))

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
        prereqs = build_evolution_prereqs(
            skill_name=skill_names[0],
            config=config,
            eval_source=eval_source,
            external_sources=list(external_sources) if external_sources else None,
            reuse_dataset=reuse_dataset,
        )
        params: SkillEvolverParams = SkillEvolverParams(
            skill_name=skill_names[0],
            eval_source=eval_source,
            external_sources=list(external_sources) if external_sources else None,
            iterations=iterations,
            reuse_dataset=reuse_dataset,
            min_improvement=min_improvement,
            prior_metrics=prereqs.prior_metrics,
            cached_path=prereqs.cached_path,
            config=config,
            console=prereqs.console,
            prebuilt_skill=prereqs.skill,
            prebuilt_dataset=prereqs.dataset,
            prebuilt_baseline_module=prereqs.baseline_module,
            prebuilt_trainset=prereqs.trainset,
            prebuilt_valset=prereqs.valset,
        )
        metrics = evolve_single_skill(params)
        _print_summary([metrics])
        return

    # ── Batch (--all) ─────────────────────────────────────────────────────────
    all_metrics = evolve_skills_batch(
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
