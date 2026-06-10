# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration — single-skill entry point.

Each numbered step delegates to the matching stage module under
skill_evolver_stages/. The logic that used to live here is now in:

  stage02_skill_constraint_validator    — validate evolved skill (stage 07)
  stage05_gepa_optimizer                — run GEPA (or MIPROv2 fallback)
  stage06_evolved_skill_extractor       — extract evolved body + reassemble
  stage07_holdout_evaluator             — score baseline vs evolved on holdout
  stage08_acceptance_gate               — apply min_improvement threshold
  stage09_results_display               — print Rich table / plain-text summary
  stage10_output_saver                  — write artifacts + metrics_history.jsonl

Stages 1–4 (skill loading, constraint validation, dataset building, DSPy
configuration) are handled by the caller via
:func:`~.skill_evolver_prereqs.build_evolution_prereqs` (production / CLI /
batch) or ``step_01_build_skill_dataset_and_dspy`` (demo pipeline).
All four artifacts are passed in via :class:`~.skill_evolver_single_params.SkillEvolverParams`.
"""
from __future__ import annotations

from datetime import datetime

from .skill_evolver_stages.stage02_skill_constraint_validator.skill_constraint_validator import validate_skill_constraints
from .skill_evolver_stages.stage05_gepa_optimizer.gepa_optimizer_runner import run_gepa_optimization
from .skill_evolver_stages.stage06_holdout_evaluator.holdout_evaluator import evaluate_on_holdout
from .skill_evolver_stages.stage07_acceptance_gates.gates_applier import apply_acceptance_gates
from .skill_evolver_stages.stage08_results_display.results_displayer import display_results_table
from .skill_evolver_stages.stage09_output_saver.output_saver import save_outputs
from .skill_evolver_single_params import SkillEvolverParams


def evolve_single_skill(params: SkillEvolverParams) -> dict:
    """Run one GEPA evolution pass on a skill (stages 5–11).

    Stages 1–4 must be completed by the caller before this function is
    invoked.  All required artifacts are read from *params*:

    Args:
        params: Fully populated :class:`SkillEvolverParams`.  The
            ``prebuilt_skill``, ``prebuilt_dataset``,
            ``prebuilt_baseline_module``, ``prebuilt_trainset``,
            ``prebuilt_valset``, ``config``, and ``console`` fields are
            mandatory (no defaults).  ``prior_metrics`` and ``cached_path``
            are optional and default to ``None``.

    Returns:
        Metrics dict including baseline_score, evolved_score, improvement,
        accepted (bool), and cross_run_delta (vs prior best if available).
    """
    params.console.print("Single Skill Evolver - Evolve Started")

    # ── Output directory for this evolution run ───────────────────────────────
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = params.config.output_dir / params.skill_name / time_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 5: Run GEPA (or MIPROv2 fallback) ───────────────────────────────
    optimized_module, optimizer_name, elapsed, evolved_text = (
        run_gepa_optimization(params.prebuilt_baseline_module,
                              params.prebuilt_skill["frontmatter_text"],
                              params.prebuilt_trainset,
                              params.prebuilt_valset,
                              params.config,
                              params.console,
                              params.skill_name))

    # ── Step 2: Validate evolved constraints ─────────────────────────────────
    evolved_checks, constraints_passed = (
        validate_skill_constraints(evolved_text,
                                   params.config,
                                   params.console,
                                   stage_label="Evolved",
                                   baseline_text=params.prebuilt_skill["raw"],
                                   output_dir=output_dir))

    # ── Step 6: Evaluate on holdout ──────────────────────────────────────────
    baseline_score, evolved_score, improvement, cross_run_delta, evolved_dims_rubrics, rubrics_state = \
        evaluate_on_holdout(optimized_module,
                            params.config.scoring_mode,
                            params.prebuilt_dataset,
                            params.config,
                            params.console,
                            params.prior_metrics,
                            params.prior_baseline_score_holistic,
                            params.prior_baseline_score_graph,
                            params.prior_baseline_dims_rubrics,
                            params.prebuilt_skill["raw"],
                            evolved_text,
                            prior_baseline_score_checklist=params.prior_baseline_score_checklist,
                            prior_baseline_score_instruction_following=params.prior_baseline_score_instruction_following,
                            prior_baseline_score_consistency=params.prior_baseline_score_consistency,
                            baseline_module=params.prebuilt_baseline_module)

    # ── Step 07: Acceptance gate (threshold or Thompson Sampling) ─────────────
    accepted, ts_conf = apply_acceptance_gates(constraints_passed, params.config, params.min_improvement,
                                               improvement, evolved_score, params.skill_name,
                                               evolved_text, cross_run_delta, output_dir, params.console)

    # ── Step 08: Display result table ───────────────────────────────────────
    display_results_table(params.skill_name,
                          optimizer_name,
                          params.config.iterations,
                          baseline_score,
                          evolved_score,
                          improvement,
                          cross_run_delta,
                          accepted,
                          elapsed,
                          len(params.prebuilt_skill["raw"]),
                          len(evolved_text),
                          params.console,
                          constraint_checks=evolved_checks,
                          prior_baseline_dims_rubrics=params.prior_baseline_dims_rubrics,
                          evolved_dims_rubrics=evolved_dims_rubrics,
                          mo_weights=rubrics_state.weights if rubrics_state is not None else None)

    # ── Step 09: Save outputs ─────────────────────────────────────────────────
    output_result = save_outputs(params.skill_name,
                                 time_stamp,
                                 baseline_score,
                                 evolved_score,
                                 improvement,
                                 accepted,
                                 params.min_improvement,
                                 cross_run_delta,
                                 params.prior_metrics,
                                 params.config,
                                 optimizer_name,
                                 params.eval_source,
                                 params.cached_path,
                                 params.prebuilt_skill["raw"],
                                 evolved_text,
                                 evolved_checks,
                                 elapsed,
                                 output_dir,
                                 params.console,
                                 ts_confidence=ts_conf)

    params.console.print("Single Skill Evolver - Evolve Finished")
    return output_result
