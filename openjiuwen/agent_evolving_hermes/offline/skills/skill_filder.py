# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DSPy SkillModule — wraps a SKILL.md as an optimisable DSPy module.

Mirrors hermes-agent-self-evolution evolution/skills/skill_module.py exactly.
"""
from pathlib import Path
from typing import Optional


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
