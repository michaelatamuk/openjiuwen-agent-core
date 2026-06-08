# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Multi-objective fitness scoring for GEPA optimisation.

Extends the existing single-scalar judge with 5 independent dimensions:
  1. correctness         — did the agent do the right thing?
  2. procedure_following — did it follow the specified workflow?
  3. conciseness         — appropriately concise?
  4. completeness        — did it cover all required aspects?
  5. specificity         — are findings specific and actionable, not vague?

The existing LLMJudge and FitnessScore are NOT modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, List

import dspy


@dataclass
class MultiRubricFitnessScore:
    correctness: float = 0.0
    procedure_following: float = 0.0
    conciseness: float = 0.0
    completeness: float = 0.0
    specificity: float = 0.0
    feedback: str = ""

    DIM_NAMES: ClassVar[List[str]] = [
        "correctness",
        "procedure_following",
        "conciseness",
        "completeness",
        "specificity",
    ]

    def as_list(self) -> List[float]:
        return [
            self.correctness,
            self.procedure_following,
            self.conciseness,
            self.completeness,
            self.specificity,
        ]


class MultiRubricsLLMJudge:
    """LLM-as-judge scorer with 5 independent quality dimensions.

    Used only when ``scoring_mode="rubrics"``.  The existing ``LLMJudge`` is
    untouched; this class is purely additive.
    """

    class MultiRubricJudgeSignature(dspy.Signature):
        """Score an agent response across five independent quality dimensions.

        Return five independent float scores (0.0–1.0) and brief feedback.
        Each dimension is scored independently — do not let one influence another.
        """

        task_input: str = dspy.InputField(desc="The task given to the agent")
        expected_behavior: str = dspy.InputField(
            desc="Rubric: what a good response looks like"
        )
        agent_output: str = dspy.InputField(desc="The actual agent response to score")
        skill_text: str = dspy.InputField(
            desc="The skill instructions the agent was given"
        )
        correctness: float = dspy.OutputField(
            desc="0.0–1.0: Did the agent do the right thing according to the task?"
        )
        procedure_following: float = dspy.OutputField(
            desc="0.0–1.0: Did the agent follow the specified workflow in the skill?"
        )
        conciseness: float = dspy.OutputField(
            desc="0.0–1.0: Was the response appropriately concise and free of padding?"
        )
        completeness: float = dspy.OutputField(
            desc="0.0–1.0: Did the response cover all required aspects of the task?"
        )
        specificity: float = dspy.OutputField(
            desc="0.0–1.0: Are findings specific and actionable rather than vague?"
        )
        feedback: str = dspy.OutputField(
            desc="One sentence identifying the main strength or most important weakness."
        )

    def __init__(self, model: str, max_skill_size: int = 15_000) -> None:
        self.judge = dspy.ChainOfThought(self.MultiRubricJudgeSignature)
        self.model = model
        self.max_skill_size = max_skill_size

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
    ) -> MultiRubricFitnessScore:
        lm = dspy.LM(self.model)
        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        return MultiRubricFitnessScore(
            correctness=float(getattr(result, "correctness", 0.5)),
            procedure_following=float(getattr(result, "procedure_following", 0.5)),
            conciseness=float(getattr(result, "conciseness", 0.5)),
            completeness=float(getattr(result, "completeness", 0.5)),
            specificity=float(getattr(result, "specificity", 0.5)),
            feedback=getattr(result, "feedback", ""),
        )
