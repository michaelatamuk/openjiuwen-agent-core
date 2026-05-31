# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Eval dataset construction for GEPA skill evolution.

Mirrors hermes-agent-self-evolution evolution/core/dataset_builder.py exactly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalExample:
    task_input: str
    expected_behavior: str
    difficulty: str = "medium"  # "easy" | "medium" | "hard"
    category: str = "general"
    source: str = "synthetic"   # "synthetic" | "golden" | "jiuwen" | "claude-code"

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
