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


# ── Skill file utilities ──────────────────────────────────────────────────────


def find_skill(name: str, skills_root: Path) -> Optional[Path]:
    """Return SKILL.md path for the named skill, or None."""
    direct = skills_root / name / "SKILL.md"
    if direct.exists():
        return direct
    for category_dir in skills_root.iterdir():
        if not category_dir.is_dir():
            continue
        candidate = category_dir / name / "SKILL.md"
        if candidate.exists():
            return candidate
    return None


def load_skill(skill_path: Path) -> Dict[str, str]:
    """Read and parse a SKILL.md file.

    Returns dict with keys: raw, frontmatter_text, frontmatter, body, name, description.
    """
    raw = skill_path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            fm_text = raw[3:end]
            body = raw[end + 4:].lstrip("\n")
            try:
                fm = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError:
                fm = {}
            return {
                "raw": raw,
                "frontmatter_text": fm_text,
                "frontmatter": fm,
                "body": body,
                "name": fm.get("name", skill_path.parent.name),
                "description": fm.get("description", ""),
            }
    return {
        "raw": raw,
        "frontmatter_text": "",
        "frontmatter": {},
        "body": raw,
        "name": skill_path.parent.name,
        "description": "",
    }


def reassemble_skill(frontmatter_text: str, evolved_body: str) -> str:
    """Re-combine frontmatter and evolved body back into a complete SKILL.md."""
    return f"---{frontmatter_text}\n---\n\n{evolved_body.lstrip()}"
