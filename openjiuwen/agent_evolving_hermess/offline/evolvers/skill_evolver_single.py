# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration — single-skill entry point.

Each numbered step delegates to the matching stage module under
skill_evolver_stages/. The logic that used to live here is now in:

  stage01_skill_finder_and_loader      — find + load SKILL.md
  stage02_baseline_constraint_validator — validate baseline before evolving
  stage03_dataset_builder               — build / reuse eval dataset
  stage04_dspy_configurator             — configure DSPy LM + splits
  stage05_gepa_optimizer                — run GEPA (or MIPROv2 fallback)
  stage06_evolved_skill_extractor       — extract evolved body + reassemble
  stage07_evolved_constraint_validator  — validate evolved skill
  stage08_holdout_evaluator             — score baseline vs evolved on holdout
  stage09_acceptance_gate               — apply min_improvement threshold
  stage10_results_display               — print Rich table / plain-text summary
  stage11_output_saver                  — write artefacts + metrics_history.jsonl
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from ._console_maker import _make_console
from .skill_evolver_stages.stage01_skill_finder_and_loader import find_and_load_skill
from .skill_evolver_stages.stage02_baseline_constraint_validator import validate_baseline_constraints
from .skill_evolver_stages.stage03_dataset_builder import build_or_load_dataset
from .skill_evolver_stages.stage04_dspy_configurator import configure_dspy_and_prepare_sets
from .skill_evolver_stages.stage05_gepa_optimizer import run_gepa_optimization
from .skill_evolver_stages.stage06_evolved_skill_extractor import extract_evolved_skill
from .skill_evolver_stages.stage07_evolved_constraint_validator import validate_evolved_constraints
from .skill_evolver_stages.stage08_holdout_evaluator import evaluate_on_holdout
from .selection import make_acceptance_gate
from .skill_evolver_stages.stage10_results_display import display_results_table
from .skill_evolver_stages.stage11_output_saver import save_outputs
from ..config import EvolverConfig


def evolve_single_skill(
    skill_name: str,
    eval_source: str = "synthetic",
    external_sources: Optional[list] = None,
    iterations: Optional[int] = None,
    config: Optional[EvolverConfig] = None,
    reuse_dataset: bool = False,
    min_improvement: float = 0.0,
) -> dict:
    """Run one GEPA evolution pass on a skill.

    Args:
        skill_name: Name of the skill to evolve.
        eval_source: "synthetic" | "external" | "golden"
        external_sources: List of external log sources for eval_source="external".
        iterations: Override config.iterations if provided.
        config: EvolverConfig instance; defaults constructed if None.
        reuse_dataset: Reuse the most recently cached dataset (saves LLM cost).
        min_improvement: Minimum fitness improvement required before the evolved
            skill is accepted.  If improvement < min_improvement, the result is
            saved as evolved_REGRESSION.md but NOT deployed to evolved_skill.md.
            Use 0.0 (default) to accept any positive improvement.
            Use a negative value (e.g. -0.05) to accept minor regressions.

    Returns:
        Metrics dict including baseline_score, evolved_score, improvement,
        accepted (bool), and cross_run_delta (vs prior best if available).
    """
    if config is None:
        config = EvolverConfig()
    if iterations is not None:
        config.iterations = iterations

    console = _make_console()

    # ── Step 1: Find and load skill ──────────────────────────────────────────
    skill, prior_metrics = find_and_load_skill(skill_name, config, console)

    # ── Step 2: Validate baseline constraints ────────────────────────────────
    validate_baseline_constraints(skill["raw"], config, console)

    # ── Step 3: Build / reuse eval dataset ───────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / skill_name / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, cached_path = build_or_load_dataset(
        skill_name, skill["raw"], eval_source, external_sources,
        config, output_dir, reuse_dataset, console,
    )

    # ── Step 4: Configure DSPy + prepare train/val sets ──────────────────────
    baseline_module, trainset, valset = configure_dspy_and_prepare_sets(
        skill["raw"], dataset, config,
    )

    # ── Step 5: Run GEPA (or MIPROv2 fallback) ───────────────────────────────
    optimized_module, optimizer_name, elapsed = run_gepa_optimization(
        baseline_module, trainset, valset, config, console,
        skill_name=skill_name,
    )

    # ── Step 6: Extract evolved skill text ───────────────────────────────────
    evolved_text = extract_evolved_skill(optimized_module, skill)

    # ── Step 7: Validate evolved constraints ─────────────────────────────────
    evolved_checks = validate_evolved_constraints(
        evolved_text, skill["raw"], config, output_dir, console,
    )

    # ── Step 8: Evaluate on holdout ──────────────────────────────────────────
    baseline_score, evolved_score, improvement, cross_run_delta = evaluate_on_holdout(
        baseline_module, optimized_module, dataset, config, console, prior_metrics,
    )

    # ── Step 9: acceptance gate (threshold or Thompson Sampling) ─────────────
    gate = make_acceptance_gate(config, min_improvement)
    accepted, ts_conf = gate.decide(
        improvement, evolved_score, skill_name,
        evolved_text, cross_run_delta, output_dir, console,
    )

    # ── Step 10: Display results table ───────────────────────────────────────
    display_results_table(
        skill_name, optimizer_name, config.iterations,
        baseline_score, evolved_score, improvement,
        cross_run_delta, accepted, elapsed,
        len(skill["raw"]), len(evolved_text), console,
    )

    # ── Step 11: Save outputs ─────────────────────────────────────────────────
    return save_outputs(
        skill_name, ts,
        baseline_score, evolved_score, improvement,
        accepted, min_improvement, cross_run_delta, prior_metrics,
        config, optimizer_name, eval_source, cached_path,
        skill["raw"], evolved_text, evolved_checks,
        elapsed, output_dir, console,
        ts_confidence=ts_conf,
    )
