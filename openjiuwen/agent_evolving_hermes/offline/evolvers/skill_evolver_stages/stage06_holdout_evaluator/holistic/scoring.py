from typing import Tuple

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from .judge import HolisticLLMJudge


def run_holistic_evaluation(module: SkillModule, holdout, config,
                            console, label: str) -> float:
    n_holdout = len(holdout)
    console.print(f"\n  [bold]Evaluating skill on holdout…[/bold] [dim]({n_holdout} examples, holistic)[/dim]")

    judge = HolisticLLMJudge(model=config.eval_model, max_skill_size=config.max_skill_size)
    n = len(holdout)
    scores = []

    for i, ex in enumerate(holdout, start=1):
        sc = 0.0
        try:
            pred = module(task_input=ex.task_input)
            s = judge.score(task_input=ex.task_input,
                            expected_behavior=ex.expected_behavior,
                            agent_output=getattr(pred, "output", ""),
                            skill_text=module._skill_text_value)
            sc = s.composite
        except Exception:
            pass
        scores.append(sc)
        # RESTORED: Detailed progress print
        console.print(f"  [{i}/{n}] {label} → {sc:.4f}")

    holistic_score = sum(scores) / len(scores) if scores else 0.0

    console.print(f"  Holdout score (holistic): {holistic_score:.4f}  ({n_holdout} examples)")
    return holistic_score


def evaluate_holistic_path(module: SkillModule, holdout, config, console,
                           prior_metrics, baseline_score) -> Tuple:
    evolved_score = run_holistic_evaluation(module, holdout, config, console, "evolved")
    improvement = evolved_score - baseline_score
    delta = round(evolved_score - prior_metrics["evolved_score"], 4) if (
            prior_metrics and "evolved_score" in prior_metrics) else None

    return baseline_score, evolved_score, improvement, delta, None, None
