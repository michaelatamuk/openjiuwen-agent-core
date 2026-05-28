from __future__ import annotations

from typing import Dict

from openjiuwen.agent_evolving_hermess.offline.skills import SkillModule, reassemble_skill


def extract_evolved_skill(
    optimized_module: SkillModule,
    skill: Dict,
) -> str:
    """Extract the evolved skill body from the optimised module and reassemble.

    Combines the original frontmatter with the optimised body text.
    Returns the full evolved SKILL.md string ready for constraint validation.
    """
    evolved_body = optimized_module._skill_text_value
    return reassemble_skill(skill["frontmatter_text"], evolved_body)
