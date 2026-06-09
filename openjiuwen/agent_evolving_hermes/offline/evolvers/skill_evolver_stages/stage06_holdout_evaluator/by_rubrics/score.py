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

from dataclasses import dataclass, field
from typing import ClassVar, List

import dspy


@dataclass
class RubricsFitnessScore:
    correctness: float = 0.0
    procedure_following: float = 0.0
    format_adherence: float = 0.0
    completeness: float = 0.0
    specificity: float = 0.0
    feedback: str = ""

    DIM_NAMES: ClassVar[List[str]] = [
        "correctness",
        "procedure_following",
        "format_adherence",
        "completeness",
        "specificity",
    ]

    def as_list(self) -> List[float]:
        return [
            self.correctness,
            self.procedure_following,
            self.format_adherence,
            self.completeness,
            self.specificity,
        ]
