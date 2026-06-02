from __future__ import annotations

from typing import Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.evolvers.skill_evolver_config import EvolverConfig
from openjiuwen.agent_evolving_hermes.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from ..skill_evolver_stages.stage08_holdout_evaluator_judge import LLMJudge


def evaluate_on_holdout(
    baseline_module: SkillModule,
    optimized_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
    prior_metrics: Optional[dict] = None,
    prior_baseline_score: Optional[float] = None,
) -> Tuple[float, float, float, Optional[float]]:
    """Score the evolved module on holdout; optionally reuse a pre-computed baseline score.

    Falls back to the val split if holdout is empty.
    Returns (baseline_score, evolved_score, improvement, cross_run_delta).

    If *prior_baseline_score* is provided (already computed by an earlier
    step_01 evaluation), the baseline module is NOT re-evaluated — saving
    ~N × 20–30 s of redundant LLM calls.

    cross_run_delta is the evolved_score minus the prior run's evolved_score,
    or None if no prior run exists.
    """
    judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    holdout = dataset.holdout or dataset.val
    n_holdout = len(holdout)

    def _score(module: SkillModule, label: str) -> float:
        scores = []
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
        mean = sum(scores) / len(scores) if scores else 0.0
        return mean

    # ── Baseline ──────────────────────────────────────────────────────────────
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
        baseline_score = _score(baseline_module, "pre-train skill")
        console.print(f"  Pre-train holdout score: {baseline_score:.4f}  ({n_holdout} examples)")

    # ── Evolved ───────────────────────────────────────────────────────────────
    console.print(
        f"[bold]Evaluating evolved skill on holdout…[/bold] "
        f"[dim]({n_holdout} examples, ~25s each, no cache)[/dim]"
    )
    evolved_score = _score(optimized_module, "evolved skill")
    console.print(f"  Evolved holdout score:   {evolved_score:.4f}  ({n_holdout} examples)")

    improvement = evolved_score - baseline_score

    cross_run_delta: Optional[float] = None
    if prior_metrics and "evolved_score" in prior_metrics:
        cross_run_delta = round(evolved_score - prior_metrics["evolved_score"], 4)

    return baseline_score, evolved_score, improvement, cross_run_delta
