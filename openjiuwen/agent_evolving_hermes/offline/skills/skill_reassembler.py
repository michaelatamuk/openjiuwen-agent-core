# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DSPy SkillModule — wraps a SKILL.md as an optimisable DSPy module.

Mirrors hermes-agent-self-evolution evolution/skills/skill_module.py exactly.
"""


def reassemble_skill(frontmatter_text: str, evolved_body: str) -> str:
    """Re-combine frontmatter and evolved body back into a complete SKILL.md."""
    return f"---{frontmatter_text}\n---\n\n{evolved_body.lstrip()}"
