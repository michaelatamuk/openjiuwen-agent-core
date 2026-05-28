# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fitness scoring for GEPA optimisation.

Mirrors hermes-agent-self-evolution evolution/core/fitness.py exactly.
"""
from __future__ import annotations

import dspy


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Fast keyword-overlap metric for GEPA's inner optimisation loop.

    This is the function passed to dspy.GEPA(metric=...).
    Full LLM-as-judge is too expensive to run on every candidate.

    Mirrors Hermess skill_fitness_metric() exactly.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        return 0.0

    # Base score for non-empty output
    score = 0.5
    expected_words = set(expected.lower().split())
    output_words = set(agent_output.lower().split())

    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))
