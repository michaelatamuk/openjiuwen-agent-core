from __future__ import annotations

from typing import Dict, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule

from .scorer import ConsistencyScorer

_N_SAMPLES = 3
_TEMPERATURE = 0.7


def run_consistency_evaluation(
    module: SkillModule,
    holdout,
    config,
    console,
    label: str,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate *module* output consistency on *holdout*.

    For each example, runs the module ``_N_SAMPLES`` times at
    ``temperature=_TEMPERATURE`` and measures mean pairwise word Jaccard
    similarity between all output pairs.

    No LLM judge — purely algorithmic.

    Returns
    -------
    mean_composite : float
        Mean per-example consistency score (mean pairwise Jaccard).
    dim_means : Dict[str, float]
        ``{"mean_pairwise_similarity": mean_composite}``
    """
    n = len(holdout)
    console.print(
        f"\n  [bold]Evaluating skill consistency on holdout…[/bold] "
        f"[dim]({n} examples × {_N_SAMPLES} samples, temp={_TEMPERATURE})[/dim]"
    )

    scorer = ConsistencyScorer(n_samples=_N_SAMPLES, temperature=_TEMPERATURE)
    composites = []

    for i, ex in enumerate(holdout, start=1):
        try:
            cs = scorer.score(module, ex.task_input, config.eval_model)
            sc = cs.composite
        except Exception:
            sc = 0.0
        composites.append(sc)
        console.print(
            f"  [{i}/{n}] {label} → {sc:.4f}  "
            f"(mean pairwise similarity, n={_N_SAMPLES})"
        )

    mean_score = sum(composites) / len(composites) if composites else 0.0
    console.print(
        f"  Holdout score ({label}, consistency): {mean_score:.4f}  ({n} examples)"
    )
    return mean_score, {"mean_pairwise_similarity": mean_score}


def evaluate_consistency_path(
    module: SkillModule,
    holdout,
    config,
    console,
    prior_metrics,
    baseline_score: float,
) -> Tuple:
    """Evaluate evolved module on holdout using consistency scoring.

    Returns the same 6-tuple structure as the holistic path:

        (baseline_score, evolved_score, improvement, delta, None, None)
    """
    evolved_score, _ = run_consistency_evaluation(
        module, holdout, config, console, "evolved"
    )
    improvement = evolved_score - baseline_score
    delta: Optional[float] = (
        round(evolved_score - prior_metrics["evolved_score"], 4)
        if prior_metrics and "evolved_score" in prior_metrics
        else None
    )
    return baseline_score, evolved_score, improvement, delta, None, None
