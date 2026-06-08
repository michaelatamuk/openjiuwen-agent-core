# coding: utf-8
"""Evaluate the baseline skill on the holdout set WITHOUT any GEPA training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

from examples.agent_evolving_hermes.offline.offline_05_thompson_vs_baseline.demo.helpers.printer_banner import _banner
from offline import SkillModule, EvalDataset
from offline.evolvers.skill_evolver_stages.stage08_holdout_evaluator_judge_multi import MultiObjectiveLLMJudge, \
    MultiObjectiveFitnessScore
from openjiuwen.agent_evolving_hermes.offline import EvolverConfig, LLMJudge
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage01_skill_finder_and_loader import (
    find_and_load_skill,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage03_dataset_builder import (
    build_or_load_dataset,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage04_dspy_configurator import (
    configure_dspy_and_prepare_sets,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_stages.stage08_holdout_evaluator import (
    _eval_multi_pass,
)
from openjiuwen.agent_evolving_hermes.offline.evolvers.multi_objective_state import MultiObjectiveState

# Modes that require a multi-objective baseline pre-evaluation.
_MULTI_MODES = {"gepa_rubric"}


def step(
    skills_root: Path,
    skill_name: str,
    model: str,
    output_dir: Path,
    verbose: bool = False,
    run_modes: Optional[List[str]] = None,
) -> Tuple[float, Optional[Dict[str, float]]]:
    """Score the baseline skill on holdout for every scoring system that will be used.

    Evaluates the single-score baseline unconditionally.  If any mode in
    *run_modes* requires multi-objective scoring (currently ``"gepa_rubric"``),
    the 5-dimension baseline is also evaluated here so that GEPA runs never
    need to re-evaluate the baseline themselves.

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
    run_modes:
        List of mode names from config.  Drives whether multi-objective
        baseline evaluation is performed.  Pass ``None`` / ``[]`` to
        skip multi evaluation.

    Returns
    -------
    (single_score, multi_dims)
        ``single_score``  — composite holdout score in [0, 1] from the
                            single LLM judge (always computed).
        ``multi_dims``    — ``{dim: mean_score}`` from the multi-objective
                            judge, or ``None`` when no multi mode was requested.
    """
    needs_multi = bool(run_modes and _MULTI_MODES.intersection(run_modes))
    modes_label = "single + multi" if needs_multi else "single"
    _banner(f"① PRE-TRAINING — holdout evaluation ({modes_label})")
    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console(force_terminal=True, width=200)

    evolver_config = EvolverConfig(
        skills_root=skills_root,
        output_dir=output_dir,
        optimizer_model=model,
        eval_model=model,
        verbose=verbose,
    )

    # ── Load skill + build dataset (shared by both scoring paths) ─────────
    skill, _ = find_and_load_skill(skill_name, evolver_config, console)

    dataset, _ = build_or_load_dataset(
        skill_name=skill_name,
        skill_raw=skill["raw"],
        eval_source="golden",
        external_sources=None,
        config=evolver_config,
        output_dir=output_dir,
        reuse_dataset=False,
        console=console,
    )

    baseline_module, _, _ = configure_dspy_and_prepare_sets(
        skill["raw"], dataset, evolver_config, console
    )

    # ── Single-score evaluation ───────────────────────────────────────────
    judge = LLMJudge(model=model, max_skill_size=evolver_config.max_skill_size)
    holdout = dataset.holdout or dataset.val
    total = len(holdout)

    print(f"\n  Evaluating {total} holdout examples with single LLM-as-judge (~20–30s each)…")
    scores: list[float] = []
    for i, ex in enumerate(holdout, start=1):
        score = 0.0
        try:
            pred = baseline_module(task_input=ex.task_input)
            fs = judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=getattr(pred, "output", ""),
                skill_text=baseline_module._skill_text_value,
            )
            score = fs.composite
        except Exception:
            pass
        scores.append(score)
        print(f"  [{i}/{total}] pre-train (single) → {score:.4f}")

    single_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    print(f"  Pre-training holdout score (single): {single_score:.4f}  ({total} holdout examples)")

    # ── Multi-objective evaluation (only when a multi mode will run) ───────
    multi_dims: Optional[Dict[str, float]] = None
    multi_score = None
    if needs_multi:
        print()
        multi_score, multi_dims = score_multi_baseline(baseline_module, dataset, evolver_config, console)

    return single_score, multi_score, multi_dims


def score_multi_baseline(
    baseline_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate baseline skill with the multi-objective judge BEFORE GEPA.

    Call this after step 4 (baseline_module is ready) and pass the result to
    ``evaluate_on_holdout`` via ``prior_multi_baseline_dims`` so that the
    baseline is not re-evaluated after GEPA.

    Returns ``(equal_weight_composite, {dim: mean_score})``.
    """
    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)
    multi_judge = MultiObjectiveLLMJudge(
        model=config.eval_model, max_skill_size=config.max_skill_size
    )

    console.print(
        f"[bold]Evaluating pre-train skill on holdout…[/bold] "
        f"[dim]({n_holdout} examples, multi-objective, pre-GEPA)[/dim]"
    )
    _, dims = _eval_multi_pass(
        baseline_module, holdout, multi_judge, MultiObjectiveFitnessScore.DIM_NAMES,
        "pre-train skill", console,
    )
    # Use the same gated aggregate as GEPA runs so that Base-Rubric and
    # GEPA-Rubric pre-train scores are computed with identical formulas.
    fresh_state = MultiObjectiveState()
    b_list = [dims[d] for d in MultiObjectiveFitnessScore.DIM_NAMES]
    composite = fresh_state.aggregate(b_list)
    console.print(
        f"  Pre-train holdout score (multi): {composite:.4f}  ({n_holdout} examples)"
    )
    return composite, dims