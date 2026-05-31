# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DSPy SkillModule — wraps a SKILL.md as an optimisable DSPy module.

Mirrors hermes-agent-self-evolution evolution/skills/skill_module.py exactly.
"""


import re


def reassemble_skill(frontmatter_text: str, evolved_body: str) -> str:
    """Re-combine frontmatter and evolved body back into a complete SKILL.md.

    GEPA receives the full SKILL.md (including frontmatter) as `_skill_text_value`
    and may return it unchanged or with the frontmatter block preserved.
    Strip any leading YAML front-matter block from evolved_body before prepending
    the canonical frontmatter to avoid a doubled header.
    """
    body = re.sub(r"^---.*?---\s*", "", evolved_body.lstrip(), count=1, flags=re.DOTALL)
    return f"---{frontmatter_text}\n---\n\n{body.lstrip()}"
