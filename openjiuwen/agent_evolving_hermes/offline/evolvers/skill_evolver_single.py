# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration — single-skill entry point.

Each numbered step delegates to the matching stage module under
skill_evolver_stages/. The logic that used to live here is now in:

  stage05_gepa_optimizer                — run GEPA (or MIPROv2 fallback)
  stage06_evolved_skill_extractor       — extract evolved body + reassemble
  stage02_skill_constraint_validator    — validate evolved skill (stage 07)
  stage08_holdout_evaluator             — score baseline vs evolved on holdout
  stage09_acceptance_gate               — apply min_improvement threshold
  stage10_results_display               — print Rich table / plain-text summary
  stage11_output_saver                  — write artefacts + metrics_history.jsonl

Stages 1–4 (skill loading, constraint validation, dataset building, DSPy
configuration) are handled by the caller via
:func:`~.skill_evolver_prereqs.build_evolution_prereqs` (production / CLI /
batch) or ``step_01_build_skill_dataset_and_dspy`` (demo pipeline).
All four artefacts are passed in via :class:`~.skill_evolver_single_params.SkillEvolverParams`.
"""
from __future__ import annotations

from datetime import datetime

from .adaptive_rubric_weights import AdaptiveRubricWeights
from .selection import make_acceptance_gate
from .skill_evolver_stages.stage05_gepa_optimizer import run_gepa_optimization
from .skill_evolver_stages.stage06_evolved_skill_extractor import extract_evolved_skill
from .skill_evolver_stages.stage02_skill_constraint_validator.skill_constraint_validator import validate_skill_constraints
from .skill_evolver_stages.stage08_holdout_evaluator import evaluate_on_holdout
from .skill_evolver_stages.stage08_holdout_evaluator_judge_by_rubrics import RubricsFitnessScore
from .skill_evolver_stages.stage10_results_display import display_results_table
from .skill_evolver_stages.stage11_output_saver import save_outputs
from .skill_evolver_single_params import SkillEvolverParams


def evolve_single_skill(params: SkillEvolverParams) -> dict:
    """Run one GEPA evolution pass on a skill (stages 5–11).

    Stages 1–4 must be completed by the caller before this function is
    invoked.  All required artefacts are read from *params*:

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
    optimized_module, optimizer_name, elapsed = (
        run_gepa_optimization(params.prebuilt_baseline_module,
                              params.prebuilt_trainset,
                              params.prebuilt_valset,
                              params.config,
                              params.console,
                              params.skill_name))

    # ── Step 6: Extract evolved skill text ───────────────────────────────────
    evolved_text = extract_evolved_skill(optimized_module,
                                         params.prebuilt_skill,
                                         params.console)

    # ── Step 7: Validate evolved constraints ─────────────────────────────────
    evolved_checks, constraints_passed = (
        validate_skill_constraints(evolved_text,
                                   params.config,
                                   params.console,
                                   stage_label="Evolved",
                                   baseline_text=params.prebuilt_skill["raw"],
                                   output_dir=output_dir))

    # ── Step 8: Evaluate on holdout ──────────────────────────────────────────
    baseline_score, evolved_score, improvement, cross_run_delta, evolved_dims_multi = \
        evaluate_on_holdout(params.prebuilt_baseline_module,
                            optimized_module,
                            params.prebuilt_dataset,
                            params.config,
                            params.console,
                            params.prior_metrics,
                            prior_baseline_score_holistic=params.prior_baseline_score_holistic,
                            scoring_mode=params.config.scoring_mode,
                            prior_baseline_score_rubrics=params.prior_baseline_score_rubrics)

    # ── Step 8b: Multi-rubric processing ──────────────────────────────────
    multi_rubrics_state = None
    if params.config.scoring_mode == "rubrics" and evolved_dims_multi is not None:
        multi_rubrics_state_path = params.config.output_dir / "multi_rubrics_state.json"
        multi_rubrics_state = AdaptiveRubricWeights.load_or_create(multi_rubrics_state_path)
        b_list = [params.prior_baseline_dims_rubrics[d] for d in RubricsFitnessScore.DIM_NAMES]
        e_list = [evolved_dims_multi[d] for d in RubricsFitnessScore.DIM_NAMES]

        nr_passed, failed_dims = multi_rubrics_state.no_regression_passed(e_list, b_list)
        if not nr_passed:
            constraints_passed = False
            params.console.print(f"\n[red]No-regression check FAILED — "
                                 f"{', '.join(failed_dims)} dropped > 0.02 vs baseline[/red]")
        else:
            params.console.print("\n[green]No-regression check ✓  all 5 dimensions passed[/green]")

        def _length_penalty(text: str) -> float:
            _len = len(text)
            _thr = params.config.max_skill_size * 0.90
            if _len <= _thr:
                return 0.0
            return min(0.30, 0.30 * (_len - _thr) / (params.config.max_skill_size - _thr))

        evolved_score = multi_rubrics_state.aggregate(e_list, length_penalty=_length_penalty(evolved_text))
        baseline_score = multi_rubrics_state.aggregate(b_list, length_penalty=_length_penalty(params.prebuilt_skill["raw"]))
        improvement = evolved_score - baseline_score
        params.console.print(
            f"  Weighted scores (stage 8b): baseline={baseline_score:.4f}  evolved={evolved_score:.4f}"
            f"  Δ={improvement:+.4f}"
        )

        multi_rubrics_state.update_weights(e_list, b_list)
        multi_rubrics_state.save(multi_rubrics_state_path)

    # ── Step 9: Acceptance gate (threshold or Thompson Sampling) ─────────────
    if not constraints_passed:
        accepted = False
        ts_conf = None
    else:
        gate = make_acceptance_gate(params.config, params.min_improvement)
        accepted, ts_conf = gate.decide(improvement, evolved_score, params.skill_name,
                                        evolved_text, cross_run_delta, output_dir, params.console)

    # ── Step 10: Display result table ───────────────────────────────────────
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
                          prior_baseline_dims_multi=params.prior_baseline_dims_rubrics,
                          evolved_dims_multi=evolved_dims_multi,
                          mo_weights=multi_rubrics_state.weights if multi_rubrics_state is not None else None)

    # ── Step 11: Save outputs ─────────────────────────────────────────────────
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
