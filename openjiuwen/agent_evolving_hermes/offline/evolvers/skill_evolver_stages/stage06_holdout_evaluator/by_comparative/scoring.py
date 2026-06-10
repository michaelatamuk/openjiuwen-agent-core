from __future__ import annotations

from typing import Dict, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule

from .judge import ComparativeLLMJudge

# Fixed neutral baseline for comparative evaluation.
# A score of 0.5 means "tie" (evolved no better than baseline).
_COMPARATIVE_NEUTRAL = 0.5


def run_comparative_evaluation(
    baseline_module: SkillModule,
    evolved_module: SkillModule,
    holdout,
    config,
    console,
    label: str,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate *evolved_module* vs *baseline_module* using a head-to-head LLM judge.

    For each holdout example, both modules produce an output which is then
    presented to a preference LLM judge asking "which response better
    satisfies the rubric?".

    Returns
    -------
    mean_preference : float
        Mean per-example preference score in [0, 1].
        0.5 = tie;  > 0.5 = evolved wins on average.
    dim_means : Dict[str, float]
        ``{"preference": mean_preference}``
    """
    n = len(holdout)
    console.print(
        f"\n  [bold]Evaluating skill on holdout…[/bold] "
        f"[dim]({n} examples, comparative)[/dim]"
    )

    judge = ComparativeLLMJudge(model=config.eval_model)
    preferences = []

    for i, ex in enumerate(holdout, start=1):
        pref = 0.5
        try:
            baseline_pred = baseline_module(task_input=ex.task_input)
            evolved_pred = evolved_module(task_input=ex.task_input)
            baseline_out = getattr(baseline_pred, "output", "") or ""
            evolved_out = getattr(evolved_pred, "output", "") or ""
            cs = judge.score(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                baseline_output=baseline_out,
                evolved_output=evolved_out,
            )
            pref = cs.preference
        except Exception:
            pass
        preferences.append(pref)
        symbol = "▲" if pref > 0.5 else ("▼" if pref < 0.5 else "=")
        console.print(f"  [{i}/{n}] {label} → {pref:.4f}  {symbol}")

    mean_pref = sum(preferences) / len(preferences) if preferences else 0.5
    wins = sum(1 for p in preferences if p > 0.5)
    ties = sum(1 for p in preferences if p == 0.5)
    losses = n - wins - ties
    console.print(
        f"  Holdout score ({label}, comparative): {mean_pref:.4f}  "
        f"({n} examples — ▲{wins} ={ties} ▼{losses})"
    )
    return mean_pref, {"preference": mean_pref}


def evaluate_comparative_path(
    baseline_module: SkillModule,
    evolved_module: SkillModule,
    holdout,
    config,
    console,
    prior_metrics,
) -> Tuple:
    """Evaluate evolved module against baseline using head-to-head comparison.

    The "baseline score" for comparative runs is always the neutral 0.5
    (perfect tie).  An improvement of +0.1 means the evolved skill wins on
    approximately 60 % of holdout examples.

    Returns the same 6-tuple structure as the holistic path:

        (baseline_score, evolved_score, improvement, delta, None, None)
    """
    evolved_score, _ = run_comparative_evaluation(
        baseline_module, evolved_module, holdout, config, console, "evolved"
    )
    baseline_score = _COMPARATIVE_NEUTRAL
    improvement = evolved_score - baseline_score
    delta: Optional[float] = (
        round(evolved_score - prior_metrics["evolved_score"], 4)
        if prior_metrics and "evolved_score" in prior_metrics
        else None
    )
    return baseline_score, evolved_score, improvement, delta, None, None
