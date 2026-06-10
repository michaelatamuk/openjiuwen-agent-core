from __future__ import annotations

from typing import Dict, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule

from .judge import ChecklistLLMJudge


def run_checklist_evaluation(
    module: SkillModule,
    holdout,
    config,
    console,
    label: str,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate *module* on *holdout* using the checklist LLM judge.

    The judge extracts concrete binary criteria from ``expected_behavior``
    and checks which are satisfied by ``agent_output``.

    Returns
    -------
    mean_composite : float
        Mean of per-example ``ChecklistScore.composite`` (pass rate) values.
    dim_means : Dict[str, float]
        ``{"pass_rate": mean_composite}``
    """
    n = len(holdout)
    console.print(
        f"\n  [bold]Evaluating skill on holdout…[/bold] "
        f"[dim]({n} examples, checklist)[/dim]"
    )

    judge = ChecklistLLMJudge(model=config.eval_model)
    composites = []

    for i, ex in enumerate(holdout, start=1):
        sc = 0.0
        detail = ""
        try:
            pred = module(task_input=ex.task_input)
            pred_output = getattr(pred, "output", "") or ""
            score = judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                agent_output=pred_output,
            )
            sc = score.composite
            detail = f"  ({score.criteria_met}/{score.criteria_total} criteria)"
        except Exception:
            pass
        composites.append(sc)
        console.print(f"  [{i}/{n}] {label} → {sc:.4f}{detail}")

    mean_score = sum(composites) / len(composites) if composites else 0.0
    console.print(
        f"  Holdout score ({label}, checklist): {mean_score:.4f}  ({n} examples)"
    )
    return mean_score, {"pass_rate": mean_score}


def evaluate_checklist_path(
    module: SkillModule,
    holdout,
    config,
    console,
    prior_metrics,
    baseline_score: float,
) -> Tuple:
    """Evaluate evolved module on holdout using checklist scoring.

    Returns the same 6-tuple structure as the holistic path:

        (baseline_score, evolved_score, improvement, delta, None, None)
    """
    evolved_score, _ = run_checklist_evaluation(
        module, holdout, config, console, "evolved"
    )
    improvement = evolved_score - baseline_score
    delta: Optional[float] = (
        round(evolved_score - prior_metrics["evolved_score"], 4)
        if prior_metrics and "evolved_score" in prior_metrics
        else None
    )
    return baseline_score, evolved_score, improvement, delta, None, None
