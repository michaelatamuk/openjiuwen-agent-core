from __future__ import annotations

from typing import Dict, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule

from .judge import InstructionFollowingLLMJudge


def run_instruction_following_evaluation(
    module: SkillModule,
    holdout,
    config,
    console,
    label: str,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate *module* on *holdout* using the instruction-following LLM judge.

    The judge identifies explicit instructions in ``task_input`` and checks
    whether ``agent_output`` complied with each.  Content quality is
    explicitly out of scope — only behavioural instruction compliance is
    assessed.

    Returns
    -------
    mean_composite : float
        Mean of per-example ``InstructionFollowingScore.composite`` values.
    dim_means : Dict[str, float]
        ``{"compliance_rate": mean_composite}``
    """
    n = len(holdout)
    console.print(
        f"\n  [bold]Evaluating skill on holdout…[/bold] "
        f"[dim]({n} examples, instruction-following)[/dim]"
    )

    judge = InstructionFollowingLLMJudge(model=config.eval_model)
    composites = []

    for i, ex in enumerate(holdout, start=1):
        sc = 1.0  # default: no instructions found → full compliance
        detail = ""
        try:
            pred = module(task_input=ex.task_input)
            pred_output = getattr(pred, "output", "") or ""
            score = judge.score(
                task_input=ex.task_input,
                agent_output=pred_output,
            )
            sc = score.composite
            detail = (
                f"  ({score.instructions_followed}/{score.instructions_found} instructions)"
            )
        except Exception:
            pass
        composites.append(sc)
        console.print(f"  [{i}/{n}] {label} → {sc:.4f}{detail}")

    mean_score = sum(composites) / len(composites) if composites else 1.0
    console.print(
        f"  Holdout score ({label}, instruction-following): {mean_score:.4f}  ({n} examples)"
    )
    return mean_score, {"compliance_rate": mean_score}


def evaluate_instruction_following_path(
    module: SkillModule,
    holdout,
    config,
    console,
    prior_metrics,
    baseline_score: float,
) -> Tuple:
    """Evaluate evolved module on holdout using instruction-following scoring.

    Returns the same 6-tuple structure as the holistic path:

        (baseline_score, evolved_score, improvement, delta, None, None)
    """
    evolved_score, _ = run_instruction_following_evaluation(
        module, holdout, config, console, "evolved"
    )
    improvement = evolved_score - baseline_score
    delta: Optional[float] = (
        round(evolved_score - prior_metrics["evolved_score"], 4)
        if prior_metrics and "evolved_score" in prior_metrics
        else None
    )
    return baseline_score, evolved_score, improvement, delta, None, None
