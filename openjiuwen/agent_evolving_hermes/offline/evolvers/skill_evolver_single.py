# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Main GEPA evolution orchestration — single-skill entry point.

Each numbered step delegates to the matching stage module under
skill_evolver_stages/. The logic that used to live here is now in:

  stage05_gepa_optimizer                — run GEPA (or MIPROv2 fallback)
  stage06_evolved_skill_extractor       — extract evolved body + reassemble
  stage07_evolved_constraint_validator  — validate evolved skill
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

from .skill_evolver_stages.stage05_gepa_optimizer import run_gepa_optimization
from .skill_evolver_stages.stage06_evolved_skill_extractor import extract_evolved_skill
from .skill_evolver_stages.stage07_evolved_constraint_validator import validate_evolved_constraints
from .skill_evolver_stages.stage08_holdout_evaluator import evaluate_on_holdout
from .skill_evolver_stages.stage08_holdout_evaluator_judge_multi import MultiObjectiveFitnessScore
from .multi_objective_state import MultiObjectiveState
from .selection import make_acceptance_gate
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
    console = params.console
    config = params.config

    console.print("Single Skill Evolver - Evolve Started")

    if params.iterations is not None:
        config.iterations = params.iterations

    # ── Artefacts from stages 1–4 (provided by caller) ───────────────────────
    skill = params.prebuilt_skill
    prior_metrics = params.prior_metrics

    dataset = params.prebuilt_dataset
    cached_path = params.cached_path

    baseline_module = params.prebuilt_baseline_module
    trainset = params.prebuilt_trainset
    valset = params.prebuilt_valset

    # ── Output directory for this evolution run ───────────────────────────────
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir / params.skill_name / time_stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 5: Run GEPA (or MIPROv2 fallback) ───────────────────────────────
    optimized_module, optimizer_name, elapsed = run_gepa_optimization(
        baseline_module, trainset, valset, config, console, skill_name=params.skill_name
    )

    # ── Step 6: Extract evolved skill text ───────────────────────────────────
    evolved_text = extract_evolved_skill(optimized_module, skill, console)

    # ── Step 7: Validate evolved constraints ─────────────────────────────────
    evolved_checks, constraints_passed = validate_evolved_constraints(
        evolved_text, skill["raw"], config, output_dir, console
    )

    # ── Step 8: Evaluate on holdout ──────────────────────────────────────────
    baseline_score, evolved_score, improvement, cross_run_delta, evolved_dims_multi = \
        evaluate_on_holdout(baseline_module,
                            optimized_module,
                            dataset,
                            config,
                            console,
                            prior_metrics,
                            prior_baseline_score_single=params.prior_baseline_score_single,
                            scoring_mode=getattr(config, "scoring_mode", "single"),
                            prior_baseline_score_multi=params.prior_baseline_score_multi)

    # ── Step 8b: Multi-objective processing ──────────────────────────────────
    mo_state = None
    if getattr(config, "scoring_mode", "single") == "multi" and evolved_dims_multi is not None:
        mo_state_path = config.output_dir / "mo_state.json"
        mo_state = MultiObjectiveState.load_or_create(mo_state_path)
        b_list = [params.prior_baseline_dims_multi[d] for d in MultiObjectiveFitnessScore.DIM_NAMES]
        e_list = [evolved_dims_multi[d] for d in MultiObjectiveFitnessScore.DIM_NAMES]

        nr_passed, failed_dims = mo_state.no_regression_passed(e_list, b_list)
        if not nr_passed:
            constraints_passed = False
            console.print(
                f"\n[red]No-regression check FAILED — "
                f"{', '.join(failed_dims)} dropped > 0.02 vs baseline[/red]"
            )
        else:
            console.print("\n[green]No-regression check ✓  all 5 dimensions passed[/green]")

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

        mo_state.update_weights(e_list, b_list)
        mo_state.save(mo_state_path)

    # ── Step 9: Acceptance gate (threshold or Thompson Sampling) ─────────────
    if not constraints_passed:
        accepted, ts_conf = False, None
    else:
        gate = make_acceptance_gate(config, params.min_improvement)
        accepted, ts_conf = gate.decide(
            improvement, evolved_score, params.skill_name,
            evolved_text, cross_run_delta, output_dir, console,
        )

    # ── Step 10: Display results table ───────────────────────────────────────
    display_results_table(params.skill_name,
                          optimizer_name,
                          config.iterations,
                          baseline_score,
                          evolved_score,
                          improvement,
                          cross_run_delta,
                          accepted,
                          elapsed,
                          len(skill["raw"]),
                          len(evolved_text),
                          console,
                          constraint_checks=evolved_checks,
                          prior_baseline_dims_multi=params.prior_baseline_dims_multi,
                          evolved_dims_multi=evolved_dims_multi,
                          mo_weights=mo_state.weights if mo_state is not None else None)

    # ── Step 11: Save outputs ─────────────────────────────────────────────────
    output_result = save_outputs(params.skill_name,
                                 time_stamp,
                                 baseline_score,
                                 evolved_score,
                                 improvement,
                                 accepted,
                                 params.min_improvement,
                                 cross_run_delta,
                                 prior_metrics,
                                 config,
                                 optimizer_name,
                                 params.eval_source,
                                 cached_path,
                                 skill["raw"],
                                 evolved_text,
                                 evolved_checks,
                                 elapsed,
                                 output_dir,
                                 console,
                                 ts_confidence=ts_conf)

    console.print("Single Skill Evolver - Evolve Finished")
    return output_result
