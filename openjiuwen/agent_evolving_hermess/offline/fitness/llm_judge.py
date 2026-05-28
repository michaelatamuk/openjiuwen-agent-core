# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fitness scoring for GEPA optimisation.

Mirrors hermes-agent-self-evolution evolution/core/fitness.py exactly.
"""
from dataclasses import dataclass

import dspy

from openjiuwen.agent_evolving_hermess.offline import FitnessScore


class LLMJudge:
    """Full LLM-as-judge scorer.

    Used for final holdout evaluation, not during GEPA search
    (inner loop uses the cheaper keyword-overlap metric).
    """

    class JudgeSignature(dspy.Signature):
        """Score an agent response against the expected behavior rubric.

        Return three independent float scores (0.0–1.0) and brief feedback.
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
            desc="0.0–1.0: Did the agent do the right thing?"
        )
        procedure_following: float = dspy.OutputField(
            desc="0.0–1.0: Did it follow the specified workflow?"
        )
        conciseness: float = dspy.OutputField(
            desc="0.0–1.0: Was the response appropriately concise?"
        )
        feedback: str = dspy.OutputField(
            desc="One sentence explaining the main strength or weakness."
        )

    def __init__(self, model: str, max_skill_size: int = 15_000):
        self.judge = dspy.ChainOfThought(self.JudgeSignature)
        self.model = model
        self.max_skill_size = max_skill_size

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
    ) -> FitnessScore:
        lm = dspy.LM(self.model)
        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Length penalty: ramps 0 → 0.30 linearly from 90% to 100%+ of max_size
        skill_len = len(skill_text)
        threshold = self.max_skill_size * 0.90
        if skill_len <= threshold:
            length_penalty = 0.0
        else:
            overflow = (skill_len - threshold) / (self.max_skill_size - threshold)
            length_penalty = min(0.30, 0.30 * overflow)

        return FitnessScore(
            correctness=float(getattr(result, "correctness", 0.5)),
            procedure_following=float(getattr(result, "procedure_following", 0.5)),
            conciseness=float(getattr(result, "conciseness", 0.5)),
            length_penalty=length_penalty,
            feedback=getattr(result, "feedback", ""),
        )
