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
from .skill_evolver_stages.stage08_holdout_evaluator_judge_multi import MultiObjectiveFitnessScore
from .multi_objective_state import MultiObjectiveState
from .selection import make_acceptance_gate
from .skill_evolver_stages.stage10_results_display import display_results_table
from .skill_evolver_stages.stage11_output_saver import save_outputs
from .skill_evolver_config import EvolverConfig
from .skill_evolver_single_params import SkillEvolverParams


def evolve_single_skill(params: SkillEvolverParams) -> dict:
    """Run one GEPA evolution pass on a skill.

    Args:
        params: Skill Evolver Params with next values:
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
            prior_baseline_score_single: Pre-computed single-score baseline from step_01.
                When provided, stage08 skips re-evaluating the baseline module.
            prior_baseline_score_multi: Pre-computed multi-score baseline from step_01.
                When provided, stage08 skips re-evaluating the baseline module.
            prebuilt_skill: Pre-built skill dict from step_01_build_skill_dataset_and_dspy.  When provided,
                stage01 (find_and_load_skill) is skipped.  ``prior_metrics`` for
                cross-run delta is not available in this path (treated as None).
            prebuilt_dataset: Pre-built EvalDataset from step_01_build_skill_dataset_and_dspy.  When provided,
                stage03 (build_or_load_dataset) is skipped.
            prebuilt_baseline_module: Pre-built SkillModule from step_01_build_skill_dataset_and_dspy.  When
                provided together with prebuilt_trainset / prebuilt_valset,
                stage04 (configure_dspy_and_prepare_sets) is skipped.
            prebuilt_trainset: Pre-split DSPy training examples from step_01_build_skill_dataset_and_dspy.
            prebuilt_valset: Pre-split DSPy validation examples from step_01_build_skill_dataset_and_dspy.

    Returns:
        Metrics dict including baseline_score, evolved_score, improvement,
        accepted (bool), and cross_run_delta (vs prior best if available).
    """
    if params.console:
        console = params.console
    else:
        console = _make_console()

    console.print("Single Skill Evolver - Evolve Started")

    if params.config is None:
        config = EvolverConfig()
    else:
        config = params.config

    if params.iterations is not None:
        config.iterations = params.iterations

    # ── Step 1: Find and load skill ──────────────────────────────────────────
    if params.prebuilt_skill is not None:
        # Reuse pre-built skill; prior_metrics requires the output_dir which is
        # mode-specific and unavailable here — cross-run delta is skipped.
        skill = params.prebuilt_skill
        prior_metrics = None
    else:
        skill, prior_metrics = find_and_load_skill(params.skill_name, config, console)

    # ── Step 2: Validate baseline constraints ────────────────────────────────
    # Skip when prebuilt_skill was provided — step_01_build_skill_dataset_and_dspy already ran this in
    # proper sequence (after stage 01, before stage 03).
    if params.prebuilt_skill is None:
        validate_baseline_constraints(skill["raw"], config, console)

    # ── Step 3: Build / reuse eval dataset ───────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / params.skill_name / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    if params.prebuilt_dataset is not None:
        # Reuse pre-built dataset; cached_path is not needed downstream when
        # the dataset was already loaded by step_01_build_skill_dataset_and_dspy.
        dataset = params.prebuilt_dataset
        cached_path = None
    else:
        dataset, cached_path = build_or_load_dataset(
            params.skill_name, skill["raw"], params.eval_source, params.external_sources,
            config, output_dir, params.reuse_dataset, console,
        )

    # ── Step 4: Configure DSPy + prepare train/val sets ──────────────────────
    if (params.prebuilt_baseline_module is not None and params.prebuilt_trainset is not None
            and params.prebuilt_valset is not None):
        # Reuse pre-built module and splits; DSPy was already configured by
        # step_01_build_skill_dataset_and_dspy with the same model so no re-configuration is required.
        baseline_module = params.prebuilt_baseline_module
        trainset = params.prebuilt_trainset
        valset = params.prebuilt_valset
    else:
        baseline_module, trainset, valset = configure_dspy_and_prepare_sets(skill["raw"], dataset, config, console)

    # ── Step 5: Run GEPA (or MIPROv2 fallback) ───────────────────────────────
    optimized_module, optimizer_name, elapsed = run_gepa_optimization(baseline_module, trainset, valset, config,
                                                                      console, skill_name=params.skill_name)

    # ── Step 6: Extract evolved skill text ───────────────────────────────────
    evolved_text = extract_evolved_skill(optimized_module, skill, console)

    # ── Step 7: Validate evolved constraints ─────────────────────────────────
    evolved_checks, constraints_passed = validate_evolved_constraints(
        evolved_text, skill["raw"], config, output_dir, console,
    )

    # ── Step 8: Evaluate on holdout ──────────────────────────────────────────
    baseline_score, evolved_score, improvement, cross_run_delta, evolved_dims_multi = \
        evaluate_on_holdout(
            baseline_module, optimized_module, dataset, config, console, prior_metrics,
            prior_baseline_score_single=params.prior_baseline_score_single,
            scoring_mode=getattr(config, "scoring_mode", "single"),
            prior_baseline_score_multi=params.prior_baseline_score_multi,
        )

    # ── Step 8b: Multi-objective processing ──────────────────────────────────
    mo_state = None
    if getattr(config, "scoring_mode", "single") == "multi" and evolved_dims_multi is not None:
        # mo_state.json lives in config.output_dir (the mode root, e.g. output_no_ts/)
        # so dynamic weights persist and accumulate across multiple GEPA runs.
        mo_state_path = config.output_dir / "mo_state.json"
        mo_state = MultiObjectiveState.load_or_create(mo_state_path)
        b_list = [params.prior_baseline_dims_multi[d] for d in MultiObjectiveFitnessScore.DIM_NAMES]
        e_list = [evolved_dims_multi[d] for d in MultiObjectiveFitnessScore.DIM_NAMES]

        # No-regression check — reject if any dimension drops > 0.02
        nr_passed, failed_dims = mo_state.no_regression_passed(e_list, b_list)
        if not nr_passed:
            constraints_passed = False
            console.print(
                f"\n[red]No-regression check FAILED — "
                f"{', '.join(failed_dims)} dropped > 0.02 vs baseline[/red]"
            )
        else:
            console.print(
                "\n[green]No-regression check ✓  all 5 dimensions passed[/green]"
            )

        # Replace scalar aggregate with dynamic-weight aggregate.
        # Both baseline and evolved use the same weights so improvement is consistent,
        # and baseline_score stored in metrics matches the evolved_score scoring system.
        # Length penalty mirrors the holistic judge: ramps 0 → 0.30 between 90–100% of max_size.
        _max_size = getattr(config, "max_skill_size", 15_000)
        def _length_penalty(text: str) -> float:
            _len = len(text)
            _thr = _max_size * 0.90
            if _len <= _thr:
                return 0.0
            return min(0.30, 0.30 * (_len - _thr) / (_max_size - _thr))

        evolved_score = mo_state.aggregate(e_list, length_penalty=_length_penalty(evolved_text))
        baseline_score = mo_state.aggregate(b_list, length_penalty=_length_penalty(skill["raw"]))
        improvement    = evolved_score - baseline_score

        # Update weight state (always, regardless of acceptance)
        mo_state.update_weights(e_list, b_list)
        mo_state.save(mo_state_path)

    # ── Step 9: acceptance gate (threshold or Thompson Sampling) ─────────────
    if not constraints_passed:
        # Constraint failure is a normal rejection — not an error.
        # Skip the TS gate and mark as not accepted so steps 10-11 still run.
        accepted, ts_conf = False, None
    else:
        gate = make_acceptance_gate(config, params.min_improvement)
        accepted, ts_conf = gate.decide(
            improvement, evolved_score, params.skill_name,
            evolved_text, cross_run_delta, output_dir, console,
        )

    # ── Step 10: Display results table ───────────────────────────────────────
    display_results_table(
        params.skill_name, optimizer_name, config.iterations,
        baseline_score, evolved_score, improvement,
        cross_run_delta, accepted, elapsed,
        len(skill["raw"]), len(evolved_text), console,
        constraint_checks=evolved_checks,
        prior_baseline_dims_multi=params.prior_baseline_dims_multi,
        evolved_dims_multi=evolved_dims_multi,
        mo_weights=mo_state.weights if mo_state is not None else None,
    )

    # ── Step 11: Save outputs ─────────────────────────────────────────────────
    output_result = save_outputs(
        params.skill_name, ts,
        baseline_score, evolved_score, improvement,
        accepted, params.min_improvement, cross_run_delta, prior_metrics,
        config, optimizer_name, params.eval_source, cached_path,
        skill["raw"], evolved_text, evolved_checks,
        elapsed, output_dir, console,
        ts_confidence=ts_conf,
    )

    console.print("Single Skill Evolver - Evolve Finished")
    return output_result
