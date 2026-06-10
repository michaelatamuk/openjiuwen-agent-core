from __future__ import annotations

import dspy

from .judge_signature import ComparativeJudgeSignature
from .score import ComparativeScore


class ComparativeLLMJudge:
    """Head-to-head preference judge — no LLM needed for the final rollup.

    Calls the LLM once per example, asking it to compare ``baseline_output``
    and ``evolved_output`` against ``expected_behavior`` and return a
    continuous preference score in [0, 1].

    Usage
    -----
    ::

        judge = ComparativeLLMJudge(model="deepseek/deepseek-chat")
        cs = judge.score(
            task_input=ex.task_input,
            expected_behavior=ex.expected_behavior,
            baseline_output=baseline_pred.output,
            evolved_output=evolved_pred.output,
        )
        print(cs.preference, cs.reasoning)
    """

    def __init__(self, model: str):
        self._judge = dspy.ChainOfThought(ComparativeJudgeSignature)
        self._model = model

    def score(self,
              task_input: str,
              expected_behavior: str,
              baseline_output: str,
              evolved_output: str) -> ComparativeScore:
        lm = dspy.LM(self._model)
        with dspy.context(lm=lm):
            result = self._judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                baseline_output=baseline_output,
                evolved_output=evolved_output,
            )
        preference = float(getattr(result, "preference", 0.5))
        preference = min(1.0, max(0.0, preference))
        return ComparativeScore(
            preference=preference,
            reasoning=getattr(result, "reasoning", ""),
        )
