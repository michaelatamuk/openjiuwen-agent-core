from typing import List

from openjiuwen.agent_evolving_hermes.offline.skills import SkillModule
from .judge import HolisticLLMJudge


def score_module_holistic(
    module: SkillModule,
    holdout: list,
    judge: HolisticLLMJudge,
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
