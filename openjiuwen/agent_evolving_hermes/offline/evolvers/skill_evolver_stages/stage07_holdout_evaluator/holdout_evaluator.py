from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from ...adaptive_rubric_weights import AdaptiveRubricWeights
from ._judge_holistic import HolisticLLMJudge
from ._judge_by_rubrics import RubricsFitnessScore, RubricsLLMJudge


# ── Private helpers ────────────────────────────────────────────────────────────

def _score_module_holistic(
    module: SkillModule,
    holdout: list,
    judge: HolisticLLMJudge,
    n_holdout: int,
    console,
    label: str,
) -> float:
    """Score *module* on *holdout* with the single LLM judge. Returns mean composite."""
    scores: List[float] = []
    for i, ex in enumerate(holdout, start=1):
        sc = 0.0
        try:
            pred = module(task_input=ex.task_input)
            s = judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=getattr(pred, "output", ""),
                skill_text=module._skill_text_value,
            )
            sc = s.composite
        except Exception:
            pass
        scores.append(sc)
        console.print(f"  [{i}/{n_holdout}] {label} → {sc:.4f}")
    return sum(scores) / len(scores) if scores else 0.0


def _eval_rubrics_pass(
    module: SkillModule,
    holdout: list,
    multi_judge: RubricsLLMJudge,
    dim_names: List[str],
    label: str,
    console,
) -> Tuple[float, Dict[str, float]]:
    """Score *module* on *holdout* with the multi-objective judge."""
    n = len(holdout)
    dim_accum: Dict[str, List[float]] = {d: [] for d in dim_names}
    composites: List[float] = []

    for i, ex in enumerate(holdout, start=1):
        try:
            pred = module(task_input=ex.task_input)
            fs = multi_judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=getattr(pred, "output", ""),
                skill_text=module._skill_text_value,
            )
            vals = fs.as_list()
            for d, v in zip(dim_names, vals):
                dim_accum[d].append(v)
            composite = sum(vals) / len(vals)
            composites.append(composite)
            dims_str = ", ".join([f"{d}: {v:.2f}" for d, v in zip(dim_names, vals)])
            console.print(f"  [{i}/{n}] {label} → raw {composite:.4f} | ({dims_str})")
        except Exception:
            for d in dim_names:
                dim_accum[d].append(0.0)
            composites.append(0.0)
            console.print(f"  [{i}/{n}] {label} → 0.0000 (error)")

    composite_mean = sum(composites) / len(composites) if composites else 0.0
    dim_means = {
        d: sum(dim_accum[d]) / len(dim_accum[d]) if dim_accum[d] else 0.0
        for d in dim_names
    }
    return composite_mean, dim_means


# ── Public interface ───────────────────────────────────────────────────────────

def evaluate_on_holdout(
    baseline_module: SkillModule,
    optimized_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
    prior_metrics: Optional[dict] = None,
    prior_baseline_score_holistic: Optional[float] = None,
    scoring_mode: str = "holistic",
    prior_baseline_score_rubrics: Optional[float] = None,
) -> Tuple:
    """Score the evolved module on holdout.

    Returns a 5-tuple: (baseline_score, evolved_score, improvement, cross_run_delta, multi_dims).
    multi_dims is None when scoring_mode="holistic".
    Falls back to val split if holdout is empty.
    """
    console.print("\n[blue]~~~ Evolving Stage 08 - Evaluation On Holdout Started ~~~[/blue]")

    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)

    # ── HOLISTIC mode ───────────────────────────────────────────────────────────
    if scoring_mode != "rubrics":
        judge = HolisticLLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)

        if prior_baseline_score_holistic is not None:
            baseline_score = prior_baseline_score_holistic
            console.print(
                f"[dim]  Pre-train score (holistic, pre-computed): {baseline_score:.4f}"
                f"  — skipping re-evaluation[/dim]"
            )
        else:
            console.print(
                f"[bold]\nEvaluating pre-train skill on holdout…[/bold] "
                f"[dim]({n_holdout} examples, cached)[/dim]"
            )
            baseline_score = _score_module_holistic(
                baseline_module, holdout, judge, n_holdout, console, "pre-train skill"
            )
            console.print(f"  Pre-train holdout score: {baseline_score:.4f}  ({n_holdout} examples)")

        console.print(
            f"[bold]Evaluating evolved skill on holdout…[/bold] "
            f"[dim]({n_holdout} examples, ~25s each, no cache)[/dim]"
        )
        evolved_score = _score_module_holistic(
            optimized_module, holdout, judge, n_holdout, console, "evolved skill"
        )
        console.print(f"  Evolved holdout score:   {evolved_score:.4f}  ({n_holdout} examples)")

        improvement = evolved_score - baseline_score
        cross_run_delta: Optional[float] = None
        if prior_metrics and "evolved_score" in prior_metrics:
            cross_run_delta = round(evolved_score - prior_metrics["evolved_score"], 4)

        console.print("[blue]~~~ Evolving Stage 08 - Evaluation On Holdout Finished (Holistic) ~~~[/blue]")
        return baseline_score, evolved_score, improvement, cross_run_delta, None

    # ── RUBRICS mode ────────────────────────────────────────────────────────────
    rubrics_judge = RubricsLLMJudge(model=config.eval_model)
    dim_names = RubricsFitnessScore.DIM_NAMES

    if prior_baseline_score_rubrics is not None:
        console.print(f"[dim]  Pre-train score (rubrics, pre-computed): {prior_baseline_score_rubrics:.4f}"
                      f"  — skipping re-evaluation[/dim]")
    else:
        console.print(f"[bold]\nEvaluating pre-train skill on holdout…[/bold] "
                      f"[dim]({n_holdout} examples, multi-rubrics)[/dim]")
        prior_baseline_score_rubrics, _ = _eval_rubrics_pass(baseline_module,
                                                             holdout,
                                                             rubrics_judge,
                                                             dim_names,
                                                             "pre-train skill",
                                                             console)

        console.print(f"  Pre-train holdout score: {prior_baseline_score_rubrics:.4f}  ({n_holdout} examples)")

    console.print(f"[bold]Evaluating evolved skill on holdout…[/bold] "
                  f"[dim]({n_holdout} examples, ~25s each, multi-objective)[/dim]")

    evolved_composite, evolved_dims = _eval_rubrics_pass(optimized_module,
                                                         holdout,
                                                         rubrics_judge,
                                                         dim_names,
                                                         "evolved skill",
                                                         console)

    console.print(f"  Evolved holdout score:   {evolved_composite:.4f}  ({n_holdout} examples)"
                  f"  [dim](unweighted; adaptive weighting applied in stage 8b)[/dim]")

    improvement = evolved_composite - prior_baseline_score_rubrics
    cross_run_delta = None
    if prior_metrics and "evolved_score" in prior_metrics:
        cross_run_delta = round(evolved_composite - prior_metrics["evolved_score"], 4)

    console.print("[blue]~~~ Evolving Stage 08 - Evaluation On Holdout Finished (Rubrics) ~~~[/blue]")
    return prior_baseline_score_rubrics, evolved_composite, improvement, cross_run_delta, evolved_dims


def evaluate_baseline_on_holdout(
    baseline_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
    needs_rubrics: bool = False,
) -> Tuple[float, Optional[float], Optional[Dict[str, float]]]:
    """Score only the baseline (pre-GEPA) module on holdout.

    Called before any GEPA pass to establish pre-training scores.  The
    returned values are forwarded to ``evaluate_on_holdout`` as
    ``prior_baseline_score_single`` / ``prior_baseline_score_multi`` so
    the baseline is never re-evaluated during evolution.

    Returns:
        (holistic_score, multi_score, multi_dims).
        multi_score and multi_dims are None when needs_multi is False.
    """
    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)
    judge = HolisticLLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)

    holistic_score = round(
        _score_module_holistic(baseline_module, holdout, judge, n_holdout, console, "pre-train (single)"), 4
    )
    console.print(f"  Pre-train holdout score (holistic): {holistic_score:.4f}  ({n_holdout} examples)")

    if not needs_rubrics:
        return holistic_score, None, None

    multi_rubric_judge = RubricsLLMJudge(model=config.eval_model)
    console.print(
        f"[bold]\nEvaluating pre-train skill on holdout…[/bold] "
        f"[dim]({n_holdout} examples, rubrics)[/dim]"
    )
    _, rubrics_dims = _eval_rubrics_pass(
        baseline_module, holdout, multi_rubric_judge,
        RubricsFitnessScore.DIM_NAMES, "pre-train skill", console,
    )
    b_list = [rubrics_dims[d] for d in RubricsFitnessScore.DIM_NAMES]
    rubrics_score = AdaptiveRubricWeights().aggregate(b_list)
    console.print(f"  Pre-train holdout score (rubric): {rubrics_score:.4f}  ({n_holdout} examples)")
    return holistic_score, rubrics_score, rubrics_dims
