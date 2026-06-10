from __future__ import annotations

from typing import Dict, Optional, Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule

from .score import GraphScore
from .scorer import GraphScorer


def run_graph_evaluation(
    module: SkillModule,
    holdout,
    config,
    console,
    label: str,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate *module* on *holdout* using concept-graph similarity.

    No LLM calls — all scoring is algorithmic.

    Returns
    -------
    mean_composite : float
        Mean of per-example ``GraphScore.composite`` values.
    dim_means : Dict[str, float]
        ``{"node_score": ..., "edge_score": ...}`` — mean of each dimension.
    """
    n = len(holdout)
    console.print(
        f"\n  [bold]Evaluating skill on holdout…[/bold] "
        f"[dim]({n} examples, graph — no LLM)[/dim]"
    )

    scorer = GraphScorer()
    dim_accum: Dict[str, list] = {d: [] for d in GraphScore.DIM_NAMES}
    composites = []

    for i, ex in enumerate(holdout, start=1):
        try:
            pred = module(task_input=ex.task_input)
            pred_output = getattr(pred, "output", "")
            gs = scorer.score(ex.expected_behavior, pred_output)
            for d in GraphScore.DIM_NAMES:
                dim_accum[d].append(getattr(gs, d))
            composites.append(gs.composite)
            console.print(
                f"  [{i}/{n}] {label} → {gs.composite:.4f}  "
                f"(node={gs.node_score:.3f}, edge={gs.edge_score:.3f})"
            )
        except Exception:
            for d in GraphScore.DIM_NAMES:
                dim_accum[d].append(0.0)
            composites.append(0.0)
            console.print(f"  [{i}/{n}] {label} → 0.0000 (error)")

    dim_means = {
        d: sum(dim_accum[d]) / len(dim_accum[d]) if dim_accum[d] else 0.0
        for d in GraphScore.DIM_NAMES
    }
    mean_composite = sum(composites) / len(composites) if composites else 0.0
    console.print(
        f"  Holdout score ({label}, graph): {mean_composite:.4f}  ({n} examples)  "
        f"node={dim_means['node_score']:.4f}, edge={dim_means['edge_score']:.4f}"
    )
    return mean_composite, dim_means


def evaluate_graph_path(
    module: SkillModule,
    holdout,
    config,
    console,
    prior_metrics,
    baseline_score: float,
) -> Tuple:
    """Evaluate evolved module on holdout using graph similarity.

    Returns the same 6-tuple structure as the holistic path so that the
    rest of the pipeline (acceptance gate, results display, output saver)
    requires no changes:

        (baseline_score, evolved_score, improvement, delta, None, None)

    No adaptive weights, no no-regression checks, no per-dimension state
    persistence — graph similarity is a fast structural signal, not a
    multi-dimensional rubric.
    """
    evolved_score, _ = run_graph_evaluation(module, holdout, config, console, "evolved")

    improvement = evolved_score - baseline_score
    delta: Optional[float] = (
        round(evolved_score - prior_metrics["evolved_score"], 4)
        if prior_metrics and "evolved_score" in prior_metrics
        else None
    )
    return baseline_score, evolved_score, improvement, delta, None, None
