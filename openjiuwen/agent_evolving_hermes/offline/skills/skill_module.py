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


# ── Base signature that defines the field layout ────────────────────────────
# Instructions are left blank here; each SkillModule instance creates its own
# per-instance signature via .with_instructions(skill_text) so that GEPA
# optimises the *actual skill content* rather than this generic placeholder.

class _BaseTaskSig(dspy.Signature):
    """placeholder"""  # overridden per-instance via with_instructions()

    task_input: str = dspy.InputField(desc="The task or user request to execute")
    output: str = dspy.OutputField(
        desc="The agent's response after following the skill instructions"
    )


class SkillModule(dspy.Module):
    """Wraps a SKILL.md file as a DSPy module.

    The skill text is embedded as the DSPy signature ``instructions`` so that
    GEPA's reflection LM optimises the actual skill content.  After
    ``optimizer.compile()`` finishes, call ``sync_from_optimized()`` to read
    the evolved instruction back into ``_skill_text_value``.
    """

    def __init__(self, skill_text: str):
        super().__init__()
        # Create a per-instance Signature with the skill content as instructions.
        # with_instructions() returns a fresh Signature subclass each time, so
        # every SkillModule has its own independent instruction that GEPA can
        # evolve without interfering with other instances.
        sig = _BaseTaskSig.with_instructions(skill_text)
        self.skill_text = dspy.ChainOfThought(sig)
        # _skill_text_value is the authoritative text copy used by stage06/08.
        # It starts as the baseline and is updated by sync_from_optimized().
        self._skill_text_value = skill_text

    def forward(self, task_input: str) -> dspy.Prediction:
        return self.skill_text(task_input=task_input)

    def sync_from_optimized(self) -> None:
        """Copy the GEPA-optimised instruction into ``_skill_text_value``.

        GEPA modifies ``skill_text.predict.signature.instructions`` during
        ``compile()``.  Call this after compilation to make stage06 and
        stage08 see the evolved text via ``_skill_text_value``.
        """
        try:
            evolved = self.skill_text.predict.signature.instructions
            if evolved and evolved.strip():
                self._skill_text_value = evolved
        except AttributeError:
            pass  # Keep existing value if the attribute path is unavailable
