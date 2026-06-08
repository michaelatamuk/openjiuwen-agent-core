from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from ..adaptive_rubric_weights import AdaptiveRubricWeights
from ..skill_evolver_stages.stage08_holdout_evaluator_judge import LLMJudge
from ..skill_evolver_stages.stage08_holdout_evaluator_judge_multi import (
    MultiObjectiveFitnessScore,
    MultiObjectiveLLMJudge,
)

# ── Private helpers ────────────────────────────────────────────────────────────

def _score_module_single(
    module: SkillModule,
    holdout: list,
    judge: LLMJudge,
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


def _eval_multi_pass(
    module: SkillModule,
    holdout: list,
    multi_judge: MultiObjectiveLLMJudge,
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
            console.print(f"  [{i}/{n}] {label} → composite {composite:.4f} | ({dims_str})")
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
    prior_baseline_score_single: Optional[float] = None,
    scoring_mode: str = "single",
    prior_baseline_score_multi: Optional[float] = None,
) -> Tuple:
    """Score the evolved module on holdout.

    Returns a 5-tuple: (baseline_score, evolved_score, improvement, cross_run_delta, multi_dims).
    multi_dims is None when scoring_mode="single".
    Falls back to val split if holdout is empty.
    """
    console.print("\n[blue]~~~ Evolving Stage 08 - Evaluation On Holdout Started ~~~[/blue]")

    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)

    # ── SINGLE mode ───────────────────────────────────────────────────────────
    if scoring_mode != "multi":
        judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)

        if prior_baseline_score_single is not None:
            baseline_score = prior_baseline_score_single
            console.print(
                f"[dim]  Pre-train score (pre-computed): {baseline_score:.4f}"
                f"  — skipping re-evaluation[/dim]"
            )
        else:
            console.print(
                f"[bold]\nEvaluating pre-train skill on holdout…[/bold] "
                f"[dim]({n_holdout} examples, cached)[/dim]"
            )
            baseline_score = _score_module_single(
                baseline_module, holdout, judge, n_holdout, console, "pre-train skill"
            )
            console.print(f"  Pre-train holdout score: {baseline_score:.4f}  ({n_holdout} examples)")

        console.print(
            f"[bold]Evaluating evolved skill on holdout…[/bold] "
            f"[dim]({n_holdout} examples, ~25s each, no cache)[/dim]"
        )
        evolved_score = _score_module_single(
            optimized_module, holdout, judge, n_holdout, console, "evolved skill"
        )
        console.print(f"  Evolved holdout score:   {evolved_score:.4f}  ({n_holdout} examples)")

        improvement = evolved_score - baseline_score
        cross_run_delta: Optional[float] = None
        if prior_metrics and "evolved_score" in prior_metrics:
            cross_run_delta = round(evolved_score - prior_metrics["evolved_score"], 4)

        console.print("[blue]~~~ Evolving Stage 08 - Evaluation On Holdout Finished (Single) ~~~[/blue]")
        return baseline_score, evolved_score, improvement, cross_run_delta, None

    # ── MULTI mode ────────────────────────────────────────────────────────────
    multi_judge = MultiObjectiveLLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    dim_names = MultiObjectiveFitnessScore.DIM_NAMES

    if prior_baseline_score_multi is not None:
        console.print(
            f"[dim]  Pre-train score (multi, pre-computed): {prior_baseline_score_multi:.4f}"
            f"  — skipping re-evaluation[/dim]"
        )
    else:
        console.print(
            f"[bold]\nEvaluating pre-train skill on holdout…[/bold] "
            f"[dim]({n_holdout} examples, multi-objective)[/dim]"
        )
        prior_baseline_score_multi, _ = _eval_multi_pass(
            baseline_module, holdout, multi_judge, dim_names, "pre-train skill", console
        )
        console.print(f"  Pre-train holdout score: {prior_baseline_score_multi:.4f}  ({n_holdout} examples)")

    console.print(
        f"[bold]Evaluating evolved skill on holdout…[/bold] "
        f"[dim]({n_holdout} examples, ~25s each, multi-objective)[/dim]"
    )
    evolved_composite, evolved_dims = _eval_multi_pass(
        optimized_module, holdout, multi_judge, dim_names, "evolved skill", console
    )
    console.print(f"  Evolved holdout score:   {evolved_composite:.4f}  ({n_holdout} examples)")

    improvement = evolved_composite - prior_baseline_score_multi
    cross_run_delta = None
    if prior_metrics and "evolved_score" in prior_metrics:
        cross_run_delta = round(evolved_composite - prior_metrics["evolved_score"], 4)

    console.print("[blue]~~~ Evolving Stage 08 - Evaluation On Holdout Finished (Multi) ~~~[/blue]")
    return prior_baseline_score_multi, evolved_composite, improvement, cross_run_delta, evolved_dims


def evaluate_baseline_on_holdout(
    baseline_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
    needs_multi: bool = False,
) -> Tuple[float, Optional[float], Optional[Dict[str, float]]]:
    """Score only the baseline (pre-GEPA) module on holdout.

    Called before any GEPA pass to establish pre-training scores.  The
    returned values are forwarded to ``evaluate_on_holdout`` as
    ``prior_baseline_score_single`` / ``prior_baseline_score_multi`` so
    the baseline is never re-evaluated during evolution.

    Returns:
        (single_score, multi_score, multi_dims).
        multi_score and multi_dims are None when needs_multi is False.
    """
    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)
    judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)

    single_score = round(
        _score_module_single(baseline_module, holdout, judge, n_holdout, console, "pre-train (single)"), 4
    )
    console.print(f"  Pre-train holdout score (single): {single_score:.4f}  ({n_holdout} examples)")

    if not needs_multi:
        return single_score, None, None

    multi_judge = MultiObjectiveLLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    console.print(
        f"[bold]\nEvaluating pre-train skill on holdout…[/bold] "
        f"[dim]({n_holdout} examples, multi-objective, pre-GEPA)[/dim]"
    )
    _, dims = _eval_multi_pass(
        baseline_module, holdout, multi_judge,
        MultiObjectiveFitnessScore.DIM_NAMES, "pre-train skill", console,
    )
    b_list = [dims[d] for d in MultiObjectiveFitnessScore.DIM_NAMES]
    multi_score = AdaptiveRubricWeights().aggregate(b_list)
    console.print(f"  Pre-train holdout score (multi): {multi_score:.4f}  ({n_holdout} examples)")
    return single_score, multi_score, dims
