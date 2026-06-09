# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fitness scoring for GEPA optimisation.

Mirrors hermes-agent-self-evolution evolution/core/fitness.py exactly.
"""
from dataclasses import dataclass

import dspy


@dataclass
class FitnessScore:
    correctness: float = 0.0            # weight 0.50
    procedure_following: float = 0.0    # weight 0.30
    conciseness: float = 0.0            # weight 0.20
    length_penalty: float = 0.0         # ramps 0 → 0.30 between 90%–100% of max_size
    feedback: str = ""                  # Used by GEPA for reflection

    @property
    def composite(self) -> float:
        # Correctness gate: a fundamentally wrong answer must score near-zero
        # regardless of how polished it was.  Without this gate, an agent that
        # follows the skill instructions perfectly but produces a completely wrong
        # answer still scores 0.50 composite (0×0.5 + 1×0.3 + 1×0.2), which
        # looks acceptable to GEPA and prevents it from pivoting the skill.
        if self.correctness < 0.25:
            return max(0.0, self.correctness - self.length_penalty)

        raw = (0.50 * self.correctness + 0.30 * self.procedure_following + 0.20 * self.conciseness)
        return max(0.0, raw - self.length_penalty)

