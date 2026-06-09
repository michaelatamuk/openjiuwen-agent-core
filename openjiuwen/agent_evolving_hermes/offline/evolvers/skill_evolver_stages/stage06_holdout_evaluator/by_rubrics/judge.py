# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Rubrics fitness scoring for GEPA optimisation.

Extends the existing single-scalar judge with 5 independent dimensions:
  1. correctness         — did the agent do the right thing?
  2. procedure_following — did it follow the specified workflow?
  3. conciseness         — appropriately concise?
  4. completeness        — did it cover all required aspects?
  5. specificity         — are findings specific and actionable, not vague?

The existing LLMJudge and FitnessScore are NOT modified.
"""
from __future__ import annotations

import dspy

from .judge_signature import JudgeSignature
from .score import RubricsFitnessScore


class RubricsLLMJudge:
    """LLM-as-judge scorer with 5 independent quality dimensions.

    Used only when ``scoring_mode="rubrics"``.  The existing ``LLMJudge`` is
    untouched; this class is purely additive.
    """
    def __init__(self, model: str) -> None:
        self._judge = dspy.ChainOfThought(JudgeSignature)
        self._model = model

    def score(self, task_input: str, expected_behavior: str, agent_output: str, skill_text: str) -> RubricsFitnessScore:
        lm = dspy.LM(self._model)
        with dspy.context(lm=lm):
            result = self._judge(task_input=task_input,
                                 expected_behavior=expected_behavior,
                                 agent_output=agent_output,
                                 skill_text=skill_text)

        return RubricsFitnessScore(correctness=float(getattr(result, "correctness", 0.5)),
                                   procedure_following=float(getattr(result, "procedure_following", 0.5)),
                                   format_adherence=float(getattr(result, "format_adherence", 0.5)),
                                   completeness=float(getattr(result, "completeness", 0.5)),
                                   specificity=float(getattr(result, "specificity", 0.5)),
                                   feedback=getattr(result, "feedback", ""))
