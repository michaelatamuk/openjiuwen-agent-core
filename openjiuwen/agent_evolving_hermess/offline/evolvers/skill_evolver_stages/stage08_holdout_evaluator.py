from __future__ import annotations

from typing import Optional, Tuple

from openjiuwen.agent_evolving_hermess.offline.config import EvolverConfig
from openjiuwen.agent_evolving_hermess.offline.dataset_builder import EvalDataset
from openjiuwen.agent_evolving_hermess.offline.skills import SkillModule
from ..skill_evolver_stages.stage08_holdout_evaluator_judge import LLMJudge


def evaluate_on_holdout(
    baseline_module: SkillModule,
    optimized_module: SkillModule,
    dataset: EvalDataset,
    config: EvolverConfig,
    console,
    prior_metrics: Optional[dict] = None,
) -> Tuple[float, float, float, Optional[float]]:
    """Score both modules on the holdout split using LLMJudge.

    Falls back to the val split if holdout is empty.
    Returns (baseline_score, evolved_score, improvement, cross_run_delta).
    cross_run_delta is the evolved_score minus the prior run's evolved_score,
    or None if no prior run exists.
    """
    judge = LLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    holdout = dataset.holdout or dataset.val

    def _score(module: SkillModule) -> float:
        scores = []
        for ex in holdout:
            try:
                pred = module(task_input=ex.task_input)
                s = judge.score(
                    task_input=ex.task_input,
                    expected_behavior=ex.expected_behavior,
                    agent_output=getattr(pred, "output", ""),
                    skill_text=module._skill_text_value,
                )
                scores.append(s.composite)
            except Exception:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    console.print("[bold]Evaluating baseline on holdout…[/bold]")
    baseline_score = _score(baseline_module)
    console.print("[bold]Evaluating evolved on holdout…[/bold]")
    evolved_score = _score(optimized_module)
    improvement = evolved_score - baseline_score

    cross_run_delta: Optional[float] = None
    if prior_metrics and "evolved_score" in prior_metrics:
        cross_run_delta = round(evolved_score - prior_metrics["evolved_score"], 4)

    return baseline_score, evolved_score, improvement, cross_run_delta
