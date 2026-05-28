# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Constraint validation for evolved skill candidates.

Mirrors hermes-agent-self-evolution evolution/core/constraints.py exactly.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConstraintResult:
    passed: bool
    constraint_name: str
    message: str
