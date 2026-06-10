from __future__ import annotations

import dspy

from .judge_signature import ChecklistJudgeSignature
from .score import ChecklistScore


class ChecklistLLMJudge:
    """Checklist LLM judge — scores by binary criterion satisfaction.

    Prompts the LLM to extract concrete pass/fail criteria from
    ``expected_behavior``, then check which are satisfied by
    ``agent_output``.

    Usage
    -----
    ::

        judge = ChecklistLLMJudge(model="deepseek/deepseek-chat")
        cs = judge.score(
            task_input=ex.task_input,
            expected_behavior=ex.expected_behavior,
            agent_output=pred.output,
        )
        print(cs.composite, cs.failed_criteria)
    """

    def __init__(self, model: str):
        self._judge = dspy.ChainOfThought(ChecklistJudgeSignature)
        self._model = model

    def score(self,
              task_input: str,
              expected_behavior: str,
              agent_output: str) -> ChecklistScore:
        lm = dspy.LM(self._model)
        with dspy.context(lm=lm):
            result = self._judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
            )
        criteria_total = max(1, int(getattr(result, "criteria_total", 1)))
        criteria_met = max(0, min(criteria_total, int(getattr(result, "criteria_met", 0))))
        return ChecklistScore(
            criteria_met=criteria_met,
            criteria_total=criteria_total,
            failed_criteria=getattr(result, "failed_criteria", ""),
            feedback=getattr(result, "feedback", ""),
        )
