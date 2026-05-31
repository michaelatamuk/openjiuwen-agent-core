# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DSPy SkillModule — wraps a SKILL.md as an optimisable DSPy module.

Mirrors hermes-agent-self-evolution evolution/skills/skill_module.py exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import dspy
import yaml


class SkillModule(dspy.Module):
    """Wraps a SKILL.md file as a DSPy module.

    The `skill_text` field is the GEPA-optimisable parameter.
    `forward()` runs the skill + task through the LLM and returns the output.
    """

    class TaskWithSkill(dspy.Signature):
        """Execute a task using the provided skill instructions."""

        skill_instructions: str = dspy.InputField(
            desc="The skill instructions (full SKILL.md text) to follow"
        )
        task_input: str = dspy.InputField(desc="The task or user request to execute")
        output: str = dspy.OutputField(
            desc="The agent's response after following the skill instructions"
        )

    def __init__(self, skill_text: str):
        super().__init__()
        self.skill_text = dspy.ChainOfThought(self.TaskWithSkill)
        # Store the current skill text as a mutable parameter GEPA can mutate
        self._skill_text_value = skill_text

    def forward(self, task_input: str) -> dspy.Prediction:
        return self.skill_text(
            skill_instructions=self._skill_text_value,
            task_input=task_input,
        )
