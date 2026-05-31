# coding: utf-8
"""Evaluate the baseline skill on the holdout set WITHOUT any GEPA training."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from offline import EvolverConfig, LLMJudge
from offline.evolvers.skill_evolver_stages.stage01_skill_finder_and_loader import (
    find_and_load_skill,
)
from offline.evolvers.skill_evolver_stages.stage03_dataset_builder import (
    build_or_load_dataset,
)
from offline.evolvers.skill_evolver_stages.stage04_dspy_configurator import (
    configure_dspy_and_prepare_sets,
)


def step(skills_root: Path, skill_name: str, model: str,
         output_dir: Path, verbose: bool = False) -> float:
    """Score the *current* skill on holdout; return the composite score.

    No GEPA optimisation is performed — only the LLM-as-judge evaluation
    that GEPA would normally run as its very first action (measuring the
    baseline before any training).

    Parameters
    ----------
    skills_root:
        Root directory that contains the skill sub-directory.
    skill_name:
        Skill identifier (sub-directory name under *skills_root*).
    model:
        DSPy model string used for both running the skill and judging.
    output_dir:
        Directory where the golden dataset cache is written (reused by
        subsequent GEPA runs to avoid rebuilding).
    verbose:
        ``True`` → show DSPy / Rich INFO logs during evaluation.

    Returns
    -------
    float
        Composite holdout score in [0, 1].
    """
    _banner("⓪ BASELINE — holdout evaluation (no training)")
    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console(quiet=not verbose)

    config = EvolverConfig(
        skills_root=skills_root,
        output_dir=output_dir,
        optimizer_model=model,
        eval_model=model,
        verbose=verbose,
    )

    # ── Load skill from disk (written by step_01) ─────────────────────────
    skill, _ = find_and_load_skill(skill_name, config, console)

    # ── Build golden dataset (saved to output_dir/dataset/) ───────────────
    dataset, _ = build_or_load_dataset(
        skill_name=skill_name,
        skill_raw=skill["raw"],
        eval_source="golden",
        external_sources=None,
        config=config,
        output_dir=output_dir,
        reuse_dataset=False,
        console=console,
    )

    # ── Instantiate baseline DSPy module ──────────────────────────────────
    baseline_module, _, _ = configure_dspy_and_prepare_sets(
        skill["raw"], dataset, config
    )

    # ── Score on holdout (single pass — no "evolved" module needed) ───────
    judge = LLMJudge(model=model, max_skill_size=config.max_skill_size)
    holdout = dataset.holdout or dataset.val

    scores: list[float] = []
    for ex in holdout:
        try:
            pred = baseline_module(task_input=ex.task_input)
            fs = judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=getattr(pred, "output", ""),
                skill_text=baseline_module._skill_text_value,
            )
            scores.append(fs.composite)
        except Exception:
            scores.append(0.0)

    baseline_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    print(f"\n  Baseline holdout score: {baseline_score:.4f}  "
          f"({len(scores)} holdout examples)")
    return baseline_score
