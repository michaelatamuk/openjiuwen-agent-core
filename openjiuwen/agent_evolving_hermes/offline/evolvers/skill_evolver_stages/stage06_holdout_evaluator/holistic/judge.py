# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fitness scoring for GEPA optimisation.

Mirrors hermes-agent-self-evolution evolution/core/fitness.py exactly.
"""
from dataclasses import dataclass

import dspy

from .judge_signature import JudgeSignature
from .score import FitnessScore


class HolisticLLMJudge:
    """Full LLM-as-judge scorer.

    Used for final holdout evaluation, not during GEPA search
    (inner loop uses the cheaper keyword-overlap metric).
    """
    def __init__(self, model: str, max_skill_size: int = 15_000):
        self._judge = dspy.ChainOfThought(JudgeSignature)
        self._model = model
        self._max_skill_size = max_skill_size

    def score(self, task_input: str, expected_behavior: str, agent_output: str, skill_text: str) -> FitnessScore:
        lm = dspy.LM(self._model)
        with dspy.context(lm=lm):
            result = self._judge(task_input=task_input,
                                 expected_behavior=expected_behavior,
                                 agent_output=agent_output,
                                 skill_text=skill_text)

        # Length penalty: ramps 0 → 0.30 linearly from 90% to 100%+ of max_size
        skill_len = len(skill_text)
        threshold = self._max_skill_size * 0.90
        if skill_len <= threshold:
            length_penalty = 0.0
        else:
            overflow = (skill_len - threshold) / (self._max_skill_size - threshold)
            length_penalty = min(0.30, 0.30 * overflow)

        return FitnessScore(correctness=float(getattr(result, "correctness", 0.5)),
                            procedure_following=float(getattr(result, "procedure_following", 0.5)),
                            conciseness=float(getattr(result, "conciseness", 0.5)),
                            length_penalty=length_penalty,
                            feedback=getattr(result, "feedback", ""))
