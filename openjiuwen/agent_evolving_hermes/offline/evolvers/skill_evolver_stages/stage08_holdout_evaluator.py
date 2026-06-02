from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from ..skill_evolver_stages.stage08_holdout_evaluator_judge import LLMJudge
from ..skill_evolver_stages.stage08_holdout_evaluator_judge_multi import (
    MultiObjectiveFitnessScore,
    MultiObjectiveLLMJudge,
)


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

            # Create a string representation of the dimensions for this pass
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


def evaluate_on_holdout(
    baseline_module: SkillModule,
    optimized_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
    prior_metrics: Optional[dict] = None,
    prior_baseline_score: Optional[float] = None,
    scoring_mode: str = "single",
    prior_multi_baseline_dims: Optional[Dict[str, float]] = None,
) -> Tuple:
    """Score the evolved module on holdout.

    Returns a 5-tuple in all cases:
        (baseline_score, evolved_score, improvement, cross_run_delta, multi_scores)

    *multi_scores* is ``None`` when ``scoring_mode="single"``.
    When ``scoring_mode="multi"`` it is a dict::

        {
            "baseline": {"correctness": float, ...},
            "evolved":  {"correctness": float, ...},
        }

    ``prior_baseline_score``      — honoured in "single" mode; skips re-evaluation.
    ``prior_multi_baseline_dims`` — honoured in "multi" mode; skips re-evaluation.

    Falls back to the val split if holdout is empty.
    """
    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)

    # ── SINGLE mode ─────────────────────────────────────────────────────────
    if scoring_mode != "multi":
        judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)

        def _score_single(module: SkillModule, label: str) -> float:
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

        if prior_baseline_score is not None:
            baseline_score = prior_baseline_score
            console.print(
                f"[dim]  Pre-train score (pre-computed): {baseline_score:.4f}"
                f"  — skipping re-evaluation[/dim]"
            )
        else:
            console.print(
                f"[bold]Evaluating pre-train skill on holdout…[/bold] "
                f"[dim]({n_holdout} examples, cached)[/dim]"
            )
            baseline_score = _score_single(baseline_module, "pre-train skill")
            console.print(
                f"  Pre-train holdout score: {baseline_score:.4f}  ({n_holdout} examples)"
            )

        console.print(
            f"[bold]Evaluating evolved skill on holdout…[/bold] "
            f"[dim]({n_holdout} examples, ~25s each, no cache)[/dim]"
        )
        evolved_score = _score_single(optimized_module, "evolved skill")
        console.print(
            f"  Evolved holdout score:   {evolved_score:.4f}  ({n_holdout} examples)"
        )

        improvement = evolved_score - baseline_score
        cross_run_delta: Optional[float] = None
        if prior_metrics and "evolved_score" in prior_metrics:
            cross_run_delta = round(evolved_score - prior_metrics["evolved_score"], 4)

        return baseline_score, evolved_score, improvement, cross_run_delta, None

    # ── MULTI mode ────────────────────────────────────────────────────────────
    multi_judge = MultiObjectiveLLMJudge(
        model=config.eval_model, max_skill_size=config.max_skill_size
    )
    dim_names = MultiObjectiveFitnessScore.DIM_NAMES

    # Baseline: use pre-computed dims
    # if available, otherwise fall back to evaluating here.
    if prior_multi_baseline_dims is not None:
        baseline_dims = prior_multi_baseline_dims
        baseline_composite = sum(baseline_dims.values()) / len(baseline_dims)
        console.print(
            f"[dim]  Pre-train score (multi, pre-computed): {baseline_composite:.4f}"
            f"  — skipping re-evaluation[/dim]"
        )
    else:
        console.print(
            f"[bold]Evaluating pre-train skill on holdout…[/bold] "
            f"[dim]({n_holdout} examples, multi-objective)[/dim]"
        )
        baseline_composite, baseline_dims = _eval_multi_pass(
            baseline_module, holdout, multi_judge, dim_names, "pre-train skill", console
        )
        console.print(
            f"  Pre-train holdout score: {baseline_composite:.4f}  ({n_holdout} examples)"
        )

    console.print(
        f"[bold]Evaluating evolved skill on holdout…[/bold] "
        f"[dim]({n_holdout} examples, ~25s each, multi-objective)[/dim]"
    )
    evolved_composite, evolved_dims = _eval_multi_pass(
        optimized_module, holdout, multi_judge, dim_names, "evolved skill", console
    )
    console.print(
        f"  Evolved holdout score:   {evolved_composite:.4f}  ({n_holdout} examples)"
    )

    improvement = evolved_composite - baseline_composite

    cross_run_delta = None
    if prior_metrics and "evolved_score" in prior_metrics:
        cross_run_delta = round(evolved_composite - prior_metrics["evolved_score"], 4)

    multi_scores: Dict[str, Dict[str, float]] = {
        "baseline": baseline_dims,
        "evolved": evolved_dims,
    }

    return baseline_composite, evolved_composite, improvement, cross_run_delta, multi_scores
