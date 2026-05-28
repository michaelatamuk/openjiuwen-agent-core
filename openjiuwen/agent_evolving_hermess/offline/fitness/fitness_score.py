# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Fitness scoring for GEPA optimisation.

Mirrors hermes-agent-self-evolution evolution/core/fitness.py exactly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FitnessScore:
    correctness: float = 0.0            # weight 0.50
    procedure_following: float = 0.0    # weight 0.30
    conciseness: float = 0.0            # weight 0.20
    length_penalty: float = 0.0         # ramps 0 → 0.30 between 90%–100% of max_size
    feedback: str = ""                  # Used by GEPA for reflection

    @property
    def composite(self) -> float:
        raw = (
            0.50 * self.correctness
            + 0.30 * self.procedure_following
            + 0.20 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)
